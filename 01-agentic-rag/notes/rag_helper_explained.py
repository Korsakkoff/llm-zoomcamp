# Instrucciones del sistema para el LLM.
# Le dicen cuál es su tarea y cómo debe comportarse si no encuentra la respuesta.
INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''


# Plantilla del prompt final que recibirá el modelo.
#
# En RAG clásico, el modelo no busca por sí mismo.
# Primero Python recupera documentos relevantes.
# Luego Python construye un prompt con:
# - la pregunta del usuario
# - el contexto recuperado
PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:
    """
    Clase que agrupa los componentes de un RAG clásico.

    Un RAG clásico tiene tres fases principales:

    1. Retrieval:
       Buscar documentos relevantes para la pregunta.

    2. Augmentation:
       Construir un prompt añadiendo esos documentos como contexto.

    3. Generation:
       Enviar el prompt al LLM para generar la respuesta.

    Esta clase junta esas tres fases en un solo objeto.
    """

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model='gpt-5.4-mini'
    ):
        """
        Constructor de la clase.

        Aquí se guardan las dependencias que el RAG necesita para funcionar.

        index:
            Índice de búsqueda construido previamente sobre los documentos.
            No es el contexto directamente.
            Es la estructura que permite encontrar documentos relevantes.

        llm_client:
            Cliente de OpenAI.
            Se usa para llamar al modelo.

        instructions:
            Instrucciones generales para el comportamiento del modelo.

        prompt_template:
            Plantilla usada para construir el prompt final.

        model:
            Modelo que se usará para generar la respuesta.

        Guardar estas dependencias en self permite que todos los métodos
        de la clase puedan usarlas sin depender de variables globales.
        """

        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        """
        Fase 1: Retrieval.

        Busca documentos relevantes para la pregunta del usuario.

        query:
            Pregunta o texto de búsqueda.

        num_results:
            Número máximo de documentos que queremos recuperar.

        Internamente usa el índice:

            self.index.search(...)

        El resultado suele ser una lista de documentos, por ejemplo:

            [
                {
                    "content": "...",
                    "filename": "01-intro.md"
                },
                ...
            ]

        En RAG clásico, esta búsqueda la decide Python.
        No la decide el LLM.
        """

        return self.index.search(
            query,
            num_results=num_results
        )

    def build_context(self, search_results):
        """
        Fase 2A: Construcción del contexto.

        Convierte los documentos recuperados por search() en un bloque de texto.

        search_results:
            Lista de documentos relevantes devueltos por el índice.

        Cada documento tiene contenido y metadatos.
        Aquí añadimos:

        - doc['content']:
            El texto del documento recuperado.

        - doc['filename']:
            El nombre del archivo original.
            Esto ayuda a saber de dónde viene la información.

        El resultado final es un string largo que luego se insertará en el prompt.
        """

        lines = []

        for doc in search_results:
            lines.append(doc['content'])
            lines.append('filename: ' + doc['filename'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        """
        Fase 2B: Augmentation.

        Construye el prompt final para el LLM.

        Recibe:
        - query:
            Pregunta original del usuario.

        - search_results:
            Documentos recuperados por search().

        Primero convierte los documentos en contexto:

            context = self.build_context(search_results)

        Luego rellena la plantilla:

            QUESTION: ...
            CONTEXT: ...

        Este es el paso que convierte un LLM normal en un sistema RAG:
        el modelo no responde solo con su memoria interna, sino con contexto externo.
        """

        context = self.build_context(search_results)

        return self.prompt_template.format(
            question=query,
            context=context
        )

    def llm(self, prompt):
        """
        Fase 3: Generation.

        Envía el prompt aumentado al modelo.

        input_messages tiene dos partes:

        1. developer:
            Instrucciones generales del sistema.
            Define el comportamiento esperado.

        2. user:
            Prompt concreto con la pregunta y el contexto recuperado.

        Aquí no hay tool calling.
        El modelo no puede buscar más información.
        Solo puede responder usando el contexto que ya le hemos dado.

        Devuelve el objeto completo response, no solo el texto,
        para poder acceder también a:
        - response.output_text
        - response.usage
        - response.id
        - otros metadatos
        """

        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response

    def rag(self, query):
        """
        Pipeline completo de RAG clásico.

        Ejecuta las tres fases en orden:

        1. Retrieval:
            search_results = self.search(query)

        2. Augmentation:
            prompt = self.build_prompt(query, search_results)

        3. Generation:
            answer = self.llm(prompt)

        Finalmente devuelve una tupla:

            (
                answer.output_text,
                answer.usage
            )

        Es decir:
        - el texto de la respuesta
        - información de uso, incluyendo tokens

        Uso típico:

            answer, usage = assistant.rag("question")

            print(answer)
            print(usage.total_tokens)

        Diferencia clave con Agentic RAG:

        En este RAG clásico, Python controla el flujo:
            Python busca.
            Python construye el prompt.
            Python llama al LLM.

        En Agentic RAG, el LLM decide si quiere llamar a search,
        cuántas veces y con qué queries.
        """

        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)

        return answer.output_text, answer.usage
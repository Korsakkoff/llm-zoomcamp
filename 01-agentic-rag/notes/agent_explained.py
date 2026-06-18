from dotenv import load_dotenv
from openai import OpenAI
from ingest import load_github_data, build_index
import json
import sys

load_dotenv()

openai_client = OpenAI()

# Cargamos los documentos del curso.
# Estos documentos constituyen nuestra base de conocimiento.
documents = load_github_data()

# Construimos un índice de búsqueda sobre los documentos.
# El índice permite recuperar rápidamente contenido relevante.
index = build_index(documents)

print(f"[INIT] Loaded documents: {len(documents)}")


instructions = """
You're a course teaching assistant. Answer the student's question using the
search tool. Make multiple searches with different keywords before answering.
"""


# Definición de la herramienta que verá el modelo.
# Importante: esto NO ejecuta la función search.
# Solo describe al modelo que existe una herramienta llamada "search",
# qué hace y qué argumentos acepta.
search_tool = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

# Herramienta que el agente puede utilizar.
# Internamente consulta el índice y devuelve los documentos más relevantes.
def search(query, num_results=5):
    """
    Esta es la función real de Python.

    El modelo NO ejecuta esta función directamente.
    El modelo solo pide una llamada a herramienta.
    Luego nuestro código detecta esa petición y ejecuta esta función.
    """

    print("\n[TOOL EXECUTION]")
    print(f"Tool: search")
    print(f"Query received from model: {query!r}")
    print(f"Num results: {num_results}")

    results = index.search(
        query,
        num_results=num_results
    )

    print(f"Results returned: {len(results)}")

    # Mostramos un resumen corto de cada resultado para entender
    # qué contexto estamos devolviendo al modelo.
    for i, doc in enumerate(results, start=1):
        filename = doc.get("filename", "UNKNOWN")
        content = doc.get("content", "")
        preview = content[:180].replace("\n", " ")

        print(f"  Result #{i}")
        print(f"    filename: {filename}")
        print(f"    preview: {preview}...")

    return results


def make_call(call):
    """
    Convierte una petición de tool call del modelo en una ejecución real.

    Ejemplo:
    El modelo devuelve:
        name = "search"
        arguments = {"query": "agentic loop"}

    Python hace:
        search(query="agentic loop")

    Luego devolvemos el resultado al modelo usando function_call_output.
    """

    print("\n[MAKE CALL]")
    print(f"Requested tool name: {call.name}")
    print(f"Raw arguments: {call.arguments}")

    args = json.loads(call.arguments)

    if call.name == "search":
        result = search(**args)
    else:
        result = {
            "error": f"Unknown tool: {call.name}"
        }

    result_json = json.dumps(
        result,
        indent=2,
        default=str
    )

    print("[MAKE CALL] Returning tool output back to the model")
    print(f"Output size in characters: {len(result_json)}")

    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": result_json,
    }


def print_messages_summary(messages):
    """
    Muestra un resumen del historial que se enviará al modelo.

    El agente funciona porque conserva memoria en messages:
    - developer instructions
    - user question
    - tool calls anteriores
    - outputs de herramientas anteriores
    - respuestas parciales del modelo
    """

    print("\n[MESSAGES SUMMARY]")
    print(f"Total messages/items sent to model: {len(messages)}")

    for i, msg in enumerate(messages, start=1):
        if isinstance(msg, dict):
            role = msg.get("role", msg.get("type", "UNKNOWN"))
            content = msg.get("content", msg.get("output", ""))

            if isinstance(content, str):
                preview = content[:120].replace("\n", " ")
            else:
                preview = str(content)[:120]

            print(f"  #{i}: dict | role/type={role} | preview={preview}...")
        else:
            # Aquí suelen aparecer objetos devueltos por OpenAI:
            # function_call, message, etc.
            item_type = getattr(msg, "type", type(msg).__name__)
            print(f"  #{i}: OpenAI item | type={item_type}")


def agent_loop(
    instructions,
    question,
    model="gpt-5.4-mini",
    max_iterations=10
):
    """
    Bucle principal del agente.

    Diferencia con RAG clásico:

    RAG clásico:
        1. Python llama a search.
        2. Python construye el prompt con contexto.
        3. Python llama al LLM una vez.

    Agentic RAG:
        1. Python manda la pregunta al modelo.
        2. El modelo decide si necesita llamar a search.
        3. Python ejecuta search solo si el modelo lo pide.
        4. Python devuelve los resultados al modelo.
        5. El modelo puede pedir más búsquedas o responder.
        6. El bucle termina cuando ya no hay function_call.
    """

    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": question}
    ]

    last_answer = None
    function_call_count = 0

    print("\n[START AGENT LOOP]")
    print(f"Question: {question!r}")
    print(f"Model: {model}")
    print(f"Max iterations: {max_iterations}")

    for iteration in range(1, max_iterations + 1):
        print("\n" + "=" * 80)
        print(f"[ITERATION #{iteration}]")
        print("=" * 80)

        print_messages_summary(messages)

        print("\n[MODEL CALL]")
        print("Sending current messages + available tools to the model...")

        response = openai_client.responses.create(
            model=model,
            input=messages,
            tools=[search_tool]
        )

        print("[MODEL RESPONSE RECEIVED]")
        print(f"Response output items: {len(response.output)}")

        # Guardamos la salida del modelo en el historial.
        # Esto es clave: si el modelo pidió una tool call,
        # en la siguiente iteración debe recordar que la pidió.
        messages.extend(response.output)

        has_function_calls = False

        for item_index, item in enumerate(response.output, start=1):
            print(f"\n[OUTPUT ITEM #{item_index}]")
            print(f"Type: {item.type}")

            if item.type == "function_call":
                has_function_calls = True
                function_call_count += 1

                print("[FUNCTION CALL DETECTED]")
                print(f"Function call count so far: {function_call_count}")
                print(f"Tool name: {item.name}")
                print(f"Tool arguments: {item.arguments}")

                # Ejecutamos la herramienta pedida por el modelo
                # y añadimos el resultado al historial.
                call_output = make_call(item)
                messages.append(call_output)

            elif item.type == "message":
                print("[FINAL OR INTERMEDIATE MESSAGE DETECTED]")

                last_answer = item.content[0].text

                print("Assistant text:")
                print(last_answer)

            else:
                print("[UNKNOWN OUTPUT TYPE]")
                print(item)

        # Si el modelo no pidió ninguna herramienta en esta iteración,
        # significa que ha terminado y ha dado una respuesta final.
        if not has_function_calls:
            print("\n[STOP CONDITION]")
            print("No function calls detected. Agent loop finished.")
            break

        print("\n[CONTINUE CONDITION]")
        print("At least one function call detected. Continuing loop...")

    else:
        print("\n[MAX ITERATIONS REACHED]")
        print("The agent stopped because it reached the iteration limit.")

    print("\n[END AGENT LOOP]")
    print(f"Total function calls: {function_call_count}")

    return last_answer, function_call_count


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print('python agent.py "your question here"')
        sys.exit(1)

    # Permite ejecutar:
    # python agent.py "How does the agentic loop work?"
    #
    # También permite ejecutar sin comillas:
    # python agent.py How does the agentic loop work?
    question = " ".join(sys.argv[1:])

    answer, function_call_count = agent_loop(
        instructions=instructions,
        question=question
    )

    print("\n" + "#" * 80)
    print("FINAL RESULT")
    print("#" * 80)

    print("\nFinal answer:")
    print(answer)

    print("\nFunction calls:")
    print(function_call_count)
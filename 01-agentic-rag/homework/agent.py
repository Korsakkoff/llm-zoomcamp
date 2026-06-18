from dotenv import load_dotenv
from openai import OpenAI
from ingest import load_github_data, build_index
import json
import sys

load_dotenv()

openai_client = OpenAI()

documents = load_github_data()
index = build_index(documents)


instructions = """
You're a course teaching assistant. Answer the student's question using the
search tool. Make multiple searches with different keywords before answering.
"""


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


def search(query, num_results=5):
    return index.search(
        query,
        num_results=num_results
    )


def make_call(call):
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

    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": result_json,
    }


def agent_loop(
    instructions,
    question,
    model="gpt-5.4-mini",
    max_iterations=10
):
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": question}
    ]

    last_answer = None
    function_call_count = 0

    for iteration in range(1, max_iterations + 1):
        print(f"iteration #{iteration}...")

        response = openai_client.responses.create(
            model=model,
            input=messages,
            tools=[search_tool]
        )

        messages.extend(response.output)

        has_function_calls = False

        for item in response.output:
            if item.type == "function_call":
                has_function_calls = True
                function_call_count += 1

                print("function_call:", item.name, item.arguments)

                call_output = make_call(item)
                messages.append(call_output)

            elif item.type == "message":
                last_answer = item.content[0].text

                print("ASSISTANT:")
                print(last_answer)

        if not has_function_calls:
            break

    return last_answer, function_call_count


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print('python agent.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]

    answer, function_call_count = agent_loop(
        instructions=instructions,
        question=question
    )

    print("\nFinal answer:")
    print(answer)

    print("\nFunction calls:")
    print(function_call_count)
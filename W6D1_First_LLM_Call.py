import os
from groq import Groq


API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add your API key as an environment variable; "
        "never put it inside this file."
    )

MODEL = "llama-3.1-8b-instant"
QUESTION = "Explain what a computer network is to a beginner in a simple way."

client = Groq(api_key=API_KEY)


def ask_llm(system_prompt: str):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": QUESTION},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    usage = response.usage

    print("=" * 70)
    print(f"System prompt: {system_prompt}")
    print(f"User question: {QUESTION}")
    print("\nAssistant response:")
    print(answer)
    print("\nToken usage:")
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
    print("=" * 70)

    return response


# The same user question is sent twice.
# Only the system role changes, so the difference in style can be observed.
ask_llm(
    "You are a patient teacher. Explain concepts step by step for a complete beginner."
)

ask_llm(
    "You are a very concise expert. Answer directly using as few words as possible."
)

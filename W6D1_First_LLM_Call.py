import os
from datetime import datetime
from pathlib import Path

from groq import Groq


API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY مش موجود. حطيه كـenvironment variable ومتكتبيهوش في الكود."
    )

MODEL = "llama-3.1-8b-instant"
QUESTION = "Explain what a computer network is to a beginner in a simple way."
NOTES_FILE = Path("notes.md")

client = Groq(api_key=API_KEY)


def ask_llm(role_name, system_prompt):
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
    print(f"الدور: {role_name}")
    print(f"السؤال: {QUESTION}")
    print("\nالرد:")
    print(answer)
    print("\nاستخدام الـTokens:")
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
    print("=" * 70)

    return {
        "role": role_name,
        "answer": answer,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def save_results(results):
    with NOTES_FILE.open("a", encoding="utf-8") as file:
        file.write("\n\n---\n\n")
        file.write("# W6D1 — أول تجربة مع الـLLM\n\n")
        file.write(
            f"وقت التشغيل: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        )

        file.write("## اللي عملته\n\n")
        file.write(
            "استخدمت نفس السؤال مرتين، لكن غيرت التعليمات اللي بديها للموديل. "
            "المرة الأولى خليته يتعامل كمدرس صبور ويشرح للمبتدئ خطوة بخطوة، "
            "والمرة الثانية خليته خبير مختصر يجاوب بأقل كلام ممكن.\n\n"
        )

        file.write("## النتائج\n\n")
        file.write("| الدور | الرد | Prompt Tokens | Completion Tokens | Total Tokens |\n")
        file.write("|---|---|---:|---:|---:|\n")

        for result in results:
            answer = result["answer"].replace("\n", " ").replace("|", "\\|")
            file.write(
                f"| {result['role']} | {answer} | "
                f"{result['prompt_tokens']} | "
                f"{result['completion_tokens']} | "
                f"{result['total_tokens']} |\n"
            )

        file.write("\n## ملاحظتي\n\n")
        file.write(
            "السؤال نفسه ممكن يطلع منه رد مختلف في الأسلوب والتفصيل بمجرد "
            "ما أغير التعليمات اللي بديها للموديل. والـTokens بتوضح حجم "
            "الإدخال والإخراج بشكل رقمي، لكنها مش مقياس مباشر لجودة الإجابة.\n"
        )


def main():
    results = []

    results.append(
        ask_llm(
            "مدرس صبور",
            "You are a patient teacher. Explain concepts step by step for a complete beginner.",
        )
    )

    results.append(
        ask_llm(
            "خبير مختصر",
            "You are a very concise expert. Answer directly using as few words as possible.",
        )
    )

    # كل المطلوب العملي لليوم الأول موجود هنا.
    # النتائج والتوثيق بيتضافوا تلقائياً في notes.md.
    save_results(results)

    print("\nW6D1 خلص. النتائج اتضافت لـ notes.md")


if __name__ == "__main__":
    main()

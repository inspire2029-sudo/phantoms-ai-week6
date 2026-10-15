import os
from datetime import datetime

from groq import Groq


API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY مش موجود. حطيه كـenvironment variable ومتكتبيهوش في الكود."
    )

MODEL = "llama-3.1-8b-instant"
client = Groq(api_key=API_KEY)


def ask(question, system_prompt="جاوب بشكل مباشر. لو مش متأكد قول مش متأكد بدل ما تخمن.", temperature=0.2):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content, response.usage


def save_notes(temperature_results, hallucination_results, mitigation):
    with open("notes.md", "a", encoding="utf-8") as file:
        file.write("\n\n---\n\n")
        file.write("# W6D2 — Temperature و Hallucination\n\n")
        file.write(
            f"وقت تشغيل التجربة: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        )

        file.write("## 1. تجربة الـTemperature\n\n")
        file.write(
            "استخدمت نفس النص ونفس طلب التلخيص 3 مرات عند 0.1 و3 مرات عند 0.9. "
            "الهدف أشوف هل زيادة الـTemperature بتغير الصياغة والتفاصيل ولا لأ.\n\n"
        )
        file.write("| الحرارة | التجربة | الرد | عدد الـTokens |\n")
        file.write("|---:|---:|---|---:|\n")

        for item in temperature_results:
            answer = item["answer"].replace("\n", " ").replace("|", "\\|")
            file.write(
                f"| {item['temperature']} | {item['run']} | "
                f"{answer} | {item['tokens']} |\n"
            )

        file.write("\n### ملاحظتي\n\n")
        file.write(
            "بعد التشغيل أقارن الردود الستة. لو 0.1 و0.9 طلعوا شبه بعض جداً، "
            "يبقى أجرب موديل تاني بدل ما أفترض إن التجربة فشلت.\n\n"
        )

        file.write("## 2. اختبار الـHallucination\n\n")
        file.write("| النوع | السؤال | الرد | هل هلوس؟ | إزاي اكتشفت؟ |\n")
        file.write("|---|---|---|---|---|\n")

        for item in hallucination_results:
            answer = item["answer"].replace("\n", " ").replace("|", "\\|")
            file.write(
                f"| {item['type']} | {item['question']} | {answer} | "
                "[يتحدد بعد التحقق] | "
                f"{item['verification']} |\n"
            )

        file.write(
            "\nلو الموديل قال «معرفش» أو «مش موجود»، دي نتيجة مهمة مش فشل؛ "
            "لأن الاعتراف بعدم المعرفة أحسن من اختراع إجابة.\n\n"
        )

        file.write("## 3. تجربة تقليل الهلوسة\n\n")
        file.write(f"**السؤال:** {mitigation['question']}\n\n")
        file.write(f"**قبل تعديل الـPrompt:**\n{mitigation['before']}\n\n")
        file.write(f"**بعد تعديل الـPrompt:**\n{mitigation['after']}\n\n")
        file.write(
            "الفكرة إننا نطلب من الموديل بوضوح إنه يعترف بعدم اليقين، "
            "ونديه سياق يمنعه من اختراع معلومات. ده ممكن يقلل الهلوسة "
            "لكن مش بيلغيها.\n\n"
        )

        file.write("## 4. هل نقدر نلغي الهلوسة نهائياً؟\n\n")
        file.write(
            "لا. نقدر نقللها ونكشفها، لكن مينفعش أبني نظام حقيقي على افتراض "
            "إن الموديل عمره ما هيغلط. عملياً هستخدم مصادر موثوقة للحاجات "
            "اللي محتاجة حقائق محددة، وأخلي الموديل يقول مش متأكد بدل التخمين، "
            "وأعمل تحقق مستقل للمعلومات المهمة، خصوصاً المعلومات الحديثة. "
            "الثقة في طريقة كلام الموديل مش دليل على صحة المعلومة.\n"
        )


def main():
    # =========================
    # 1) Temperature experiment
    # =========================
    text_to_summarize = """
    الشبكات بتسمح لأجهزة مختلفة إنها تتواصل مع بعض وتبادل البيانات.
    الراوتر بيوجه البيانات بين الشبكات، والسويتش بيربط الأجهزة داخل نفس
    الشبكة المحلية. عنوان IP بيساعد في تحديد الجهاز أو الواجهة على الشبكة،
    والبروتوكولات بتحط القواعد اللي الأجهزة بتستخدمها في الاتصال. البيانات
    على الإنترنت غالباً بتتقسم لحزم صغيرة، وكل حزمة ممكن تمشي في طريق مختلف
    قبل ما توصل للوجهة. TCP بيركز على وصول البيانات بشكل موثوق وترتيبها،
    بينما UDP أبسط وممكن يكون أسرع في حالات معينة لكنه مش بيضمن نفس مستوى
    الاعتمادية. فهم الشبكات مهم في البرمجة والأمن السيبراني لأن أغلب الأنظمة
    الحديثة بتعتمد على الاتصال بين أجهزة وخدمات متعددة.
    """

    summary_prompt = (
        "لخص النص الجاي في 150 كلمة أو أكتر. حافظ على المعنى الأساسي "
        "ومتضيفش معلومات من عندك.\n\n" + text_to_summarize
    )

    temperature_results = []

    for temperature in (0.1, 0.9):
        for run in range(1, 4):
            answer, usage = ask(
                summary_prompt,
                temperature=temperature,
            )

            temperature_results.append(
                {
                    "temperature": temperature,
                    "run": run,
                    "answer": answer,
                    "tokens": usage.total_tokens,
                }
            )

    # =========================
    # 2) Hallucination experiment
    # =========================
    questions = [
        {
            "type": "معلومة حقيقية",
            "question": "ما هي عاصمة أستراليا؟",
            "verification": "أراجع مصدر موثوق عن عاصمة أستراليا.",
        },
        {
            "type": "معلومة حقيقية",
            "question": "مين كتب رواية Pride and Prejudice؟",
            "verification": "أراجع مصدر موثوق عن مؤلف الرواية.",
        },
        {
            "type": "حاجة مخترعة",
            "question": "مين مؤلف كتاب The Glass Orchard of Novera الصادر سنة 2019؟",
            "verification": "أتحقق أولاً إن الكتاب موجود فعلاً ومن مصدر مستقل.",
        },
        {
            "type": "حاجة مخترعة",
            "question": "إيه أهم خصائص مكتبة Python اسمها quantumpicklex وإمتى نزل أول إصدار مستقر ليها؟",
            "verification": "أتحقق من وجود المكتبة من مصدر رسمي أو مستودع معروف.",
        },
        {
            "type": "حدث حديث جداً",
            "question": (
                "إيه أهم حدث حصل النهارده؟ لو معندكش معلومات حديثة وموثوقة "
                "قول مش متأكد بدل ما تخمن."
            ),
            "verification": "أقارن الإجابة بمصدر أخبار موثوق بتاريخ واضح.",
        },
    ]

    hallucination_results = []

    for item in questions:
        answer, usage = ask(item["question"], temperature=0.2)

        hallucination_results.append(
            {
                **item,
                "answer": answer,
                "tokens": usage.total_tokens,
            }
        )

    # =========================
    # 3) Hallucination reduction
    # =========================
    mitigation_question = (
        "مين مؤلف كتاب The Glass Orchard of Novera الصادر سنة 2019؟"
    )

    before, _ = ask(
        mitigation_question,
        system_prompt="جاوب على السؤال بشكل مباشر.",
        temperature=0.7,
    )

    after, _ = ask(
        mitigation_question,
        system_prompt=(
            "اعتمد على معلومات موثوقة فقط. الكتاب المذكور في السؤال "
            "غير معروف لي حالياً. لو مش قادر تتحقق من وجوده أو من اسم مؤلفه، "
            "قول بوضوح «مش متأكد» وماتخترعش اسم مؤلف أو تفاصيل."
        ),
        temperature=0.2,
    )

    mitigation = {
        "question": mitigation_question,
        "before": before,
        "after": after,
    }

    # Everything required for W6D2 is handled from this one Python file.
    save_notes(
        temperature_results,
        hallucination_results,
        mitigation,
    )

    print("W6D2 خلص. النتائج اتضافت لـ notes.md")


if __name__ == "__main__":
    main()

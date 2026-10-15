import os
from datetime import datetime
from pathlib import Path

from groq import Groq


API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY مش موجود. حطي مفتاح الـAPI كـenvironment variable "
        "ومتكتبيهوش جوه الملف."
    )

MODEL = "llama-3.1-8b-instant"
NOTES_FILE = Path("notes.md")

client = Groq(api_key=API_KEY)


def ask(question, system_prompt="جاوب بشكل مباشر. لو مش متأكد من معلومة، قول إنك مش متأكد بدل ما تخمن.", temperature=0.2):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
    )

    answer = response.choices[0].message.content
    usage = response.usage
    return answer, usage


def run_temperature_test():
    text_to_summarize = """
    الشبكات بتسمح لأجهزة مختلفة إنها تتواصل مع بعض وتبادل البيانات.
    الراوتر بيوجه البيانات بين الشبكات، والسويتش بيربط الأجهزة داخل نفس الشبكة
    المحلية. عنوان IP بيساعد في تحديد الجهاز أو الواجهة على الشبكة، بينما
    البروتوكولات بتحدد القواعد اللي الأجهزة بتستخدمها في الاتصال. في الإنترنت،
    البيانات غالباً بتتقسم لحزم صغيرة، وكل حزمة ممكن تمشي في طريق مختلف قبل ما
    توصل للوجهة. بروتوكول TCP بيركز على وصول البيانات بشكل موثوق وترتيبها،
    بينما UDP أبسط وأسرع في حالات معينة ومش بيضمن نفس مستوى الاعتمادية. فهم
    الشبكات مهم في البرمجة والأمن السيبراني لأن أغلب الأنظمة الحديثة بتعتمد
    على الاتصال بين أجهزة وخدمات متعددة.
    """

    prompt = (
        "لخص النص الجاي في 150 كلمة أو أكتر. حافظ على المعنى الأساسي "
        "ومتضيفش معلومات من عندك.\n\n" + text_to_summarize
    )

    rows = []

    for temperature in (0.1, 0.9):
        for run_number in range(1, 4):
            answer, usage = ask(prompt, temperature=temperature)
            rows.append(
                {
                    "temperature": temperature,
                    "run": run_number,
                    "answer": answer,
                    "tokens": usage.total_tokens,
                }
            )

    return rows


def run_hallucination_test():
    questions = [
        {
            "type": "معلومة قابلة للتحقق",
            "question": "ما هي عاصمة أستراليا؟",
            "verification": "التحقق: كانبيرا هي العاصمة الرسمية لأستراليا.",
        },
        {
            "type": "معلومة قابلة للتحقق",
            "question": "مين كتب رواية Pride and Prejudice؟",
            "verification": "التحقق: الرواية من تأليف جين أوستن.",
        },
        {
            "type": "حاجة مش موجودة",
            "question": "مين مؤلف كتاب The Glass Orchard of Novera الصادر سنة 2019؟",
            "verification": "السؤال مبني على كتاب مخترع، فالإجابة الصحيحة المفروض تكون عدم وجود مصدر موثوق يؤكده.",
        },
        {
            "type": "حاجة مش موجودة",
            "question": "إيه أهم خصائص مكتبة Python اسمها quantumpicklex، وإمتى اتعمل أول إصدار مستقر ليها؟",
            "verification": "الاسم مستخدم هنا كاختبار لحاجة مخترعة؛ لازم نتحقق من وجود مكتبة حقيقية قبل تصديق أي تفاصيل.",
        },
        {
            "type": "حدث حديث جداً",
            "question": (
                f"إيه أهم حدث حصل النهارده {datetime.now().strftime('%Y-%m-%d')}؟ "
                "جاوب من غير استخدام أدوات بحث، ولو مش عندك معلومة موثوقة قول مش متأكد."
            ),
            "verification": "الحدث الحديث جداً لازم يتراجع من مصدر أخبار موثوق بتاريخ واضح؛ ماينفعش نعتبر ثقة النموذج دليل على صحة الإجابة.",
        },
    ]

    results = []
    for item in questions:
        answer, usage = ask(item["question"], temperature=0.2)
        results.append(
            {
                **item,
                "answer": answer,
                "tokens": usage.total_tokens,
            }
        )

    return results


def run_hallucination_reduction_test():
    question = "مين مؤلف كتاب The Glass Orchard of Novera الصادر سنة 2019؟"

    before, before_usage = ask(
        question,
        system_prompt="جاوب على السؤال بشكل مباشر.",
        temperature=0.7,
    )

    after, after_usage = ask(
        question,
        system_prompt=(
            "جاوب اعتماداً على معلومات موثوقة فقط. الكتاب المذكور في السؤال "
            "غير معروف لي حالياً. لو مش قادر تتحقق من وجوده أو من اسم مؤلفه، "
            "قول بوضوح: مش متأكد، وماتخترعش اسم مؤلف أو تفاصيل."
        ),
        temperature=0.2,
    )

    return {
        "question": question,
        "before": before,
        "before_tokens": before_usage.total_tokens,
        "after": after,
        "after_tokens": after_usage.total_tokens,
    }


def append_results_to_notes(temperature_rows, hallucination_rows, mitigation):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    with NOTES_FILE.open("a", encoding="utf-8") as file:
        file.write("\n\n---\n\n")
        file.write("# W6D2 — Temperature و Hallucination\n\n")
        file.write(f"وقت تشغيل التجربة: {timestamp}\n\n")

        file.write("## تجربة الـTemperature\n\n")
        file.write(
            "نفس النص ونفس طلب التلخيص اتبعتوا 3 مرات عند 0.1 و3 مرات عند 0.9. "
            "المهم هنا إننا نبص على اختلاف الصياغة والتفاصيل، مش بس طول الرد.\n\n"
        )
        file.write("| الحرارة | التجربة | الرد | عدد الرموز |\n")
        file.write("|---:|---:|---|---:|\n")
        for row in temperature_rows:
            compact = row["answer"].replace("\n", " ").replace("|", "\\|")
            file.write(
                f"| {row['temperature']} | {row['run']} | {compact} | {row['tokens']} |\n"
            )

        file.write("\n### ملاحظتي على الفرق\n\n")
        file.write(
            "النتيجة اللي ظهرت في التشغيل الفعلي تتسجل هنا بعد مقارنة الردود الستة. "
            "لو الردود عند 0.1 و0.9 طلعت شبه بعض جداً، ده مش معناه إن التجربة فشلت؛ "
            "ممكن يكون الموديل أو إعدادات الـAPI قللت الاختلاف، وساعتها أجرب موديل تاني.\n\n"
        )

        file.write("## اختبار الـHallucination\n\n")
        file.write("| النوع | السؤال | الرد | هل هلوس؟ | إزاي اتأكدت؟ |\n")
        file.write("|---|---|---|---|---|\n")
        for row in hallucination_rows:
            compact = row["answer"].replace("\n", " ").replace("|", "\\|")
            file.write(
                f"| {row['type']} | {row['question']} | {compact} | [يتحدد بعد التحقق] | {row['verification']} |\n"
            )

        file.write("\n### ملاحظة مهمة\n\n")
        file.write(
            "لو النموذج قال مش عارف أو قال إن الحاجة مش موجودة، ده مش فشل في التجربة. "
            "بالعكس، دي نتيجة مهمة لأنها بتوضح إن الاعتراف بعدم اليقين أأمن من اختراع إجابة.\n\n"
        )

        file.write("## تجربة تقليل الهلوسة\n\n")
        file.write(f"**السؤال:** {mitigation['question']}\n\n")
        file.write(f"**قبل تعديل الـPrompt:**\n{mitigation['before']}\n\n")
        file.write(f"**بعد تعديل الـPrompt:**\n{mitigation['after']}\n\n")
        file.write(
            "### ملاحظتي\n\n"
            "الـPrompt المعدل بيدي للموديل مساحة يقول مش متأكد، وكمان بيديه معلومة "
            "واضحة عن حدود السياق. لو النتيجة اتحسنت، ده بيقلل فرصة الهلوسة لكنه "
            "مش ضمان إن الموديل عمره ما هيغلط.\n\n"
        )

        file.write("## الخلاصة\n\n")
        file.write(
            "أنا مش شايف إننا نقدر نلغي الهلوسة نهائياً في نظام حقيقي. اللي نقدر "
            "نعمله هو نقللها ونكشفها قبل ما توصل للمستخدم. في نظام حقيقي هعتمد على "
            "مصدر موثوق أو قاعدة معرفة لما السؤال محتاج حقيقة محددة، وأطلب من "
            "الموديل يعترف بعدم اليقين بدل التخمين، وأعمل تحقق مستقل للمعلومات "
            "المهمة، خصوصاً الحاجات الحساسة أو الحديثة. كمان مش هعتبر إن نبرة "
            "الموديل الواثقة معناها إن الإجابة صح.\n"
        )


if __name__ == "__main__":
    temperature_rows = run_temperature_test()
    hallucination_rows = run_hallucination_test()
    mitigation = run_hallucination_reduction_test()

    append_results_to_notes(
        temperature_rows,
        hallucination_rows,
        mitigation,
    )

    print("W6D2 خلص. النتائج اتضافت لـ notes.md")

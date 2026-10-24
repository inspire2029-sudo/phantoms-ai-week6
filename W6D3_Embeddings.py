from pathlib import Path
from itertools import combinations

from sentence_transformers import SentenceTransformer, util


NOTES_FILE = Path("notes.md")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# الجمل دي متقسمة على 3 مجالات: كورة وبرمجة وأكل.
ENGLISH_SENTENCES = [
    # كورة
    "The striker scored the winning goal in the final minute.",
    "A late strike secured the team's victory just before the match ended.",
    "The goalkeeper stopped the penalty with an outstanding save.",
    "The keeper denied the spot kick and prevented the opponent from scoring.",
    # برمجة
    "The Python program reads data from a file and processes it.",
    "The script loads information from a file and transforms the data.",
    "The developer fixed a bug in the authentication system.",
    "The programmer repaired an error in the login mechanism.",
    # أكل
    "The chef prepared a spicy chicken dish with fresh vegetables.",
    "The cook made a hot chicken meal using fresh vegetables.",
]

# نفس كلمة bank هنا مستخدمة بمعنيين مختلفين: بنك فلوس وضفة نهر.
BANK_SENTENCES = [
    "I deposited my salary at the bank this morning.",
    "We sat on the river bank and watched the water.",
]

# Bonus: أربع جمل عربي عشان نشوف تعامل الموديل مع العربي.
ARABIC_SENTENCES = [
    "سجل المهاجم هدف الفوز في الدقيقة الأخيرة.",
    "حسم اللاعب المباراة بهدف متأخر قبل صافرة النهاية.",
    "المبرمج أصلح خطأ في نظام تسجيل الدخول.",
    "الطاهي حضر وجبة دجاج حارة بالخضروات الطازجة.",
]


# دي توقعاتي قبل ما أشوف أي نتيجة فعلية.
PREDICTIONS_TOP_3 = [
    "الجملة 1 مع الجملة 2: الاتنين بيتكلموا عن تسجيل هدف متأخر أدى للفوز.",
    "الجملة 3 مع الجملة 4: الاتنين عن حارس مرمى منع تسجيل هدف من ركلة جزاء.",
    "الجملة 5 مع الجملة 6: الاتنين عن قراءة بيانات من ملف ومعالجتها.",
]

PREDICTIONS_BOTTOM_3 = [
    "الجملة 2 مع الجملة 9: كورة مقابل أكل، فالمعنى بعيد.",
    "الجملة 4 مع الجملة 7: كورة مقابل برمجة، فالمعنى بعيد.",
    "الجملة 3 مع الجملة 10: كورة مقابل أكل، فالمعنى بعيد.",
]


def cosine_score(a, b):
    # بنستخدم دالة التشابه الجاهزة من المكتبة بدل ما نحسب المعادلة بإيدنا.
    return float(util.cos_sim(a, b)[0][0])


def pair_results(sentences, embeddings):
    results = []

    for i, j in combinations(range(len(sentences)), 2):
        results.append(
            {
                "i": i + 1,
                "j": j + 1,
                "score": cosine_score(embeddings[i], embeddings[j]),
            }
        )

    return sorted(results, key=lambda item: item["score"], reverse=True)


def print_predictions():
    print("\n=== توقعي قبل المقارنة ===")
    print("\nأعلى 3 أزواج متوقعة:")
    for prediction in PREDICTIONS_TOP_3:
        print(f"- {prediction}")

    print("\nأقل 3 أزواج متوقعة:")
    for prediction in PREDICTIONS_BOTTOM_3:
        print(f"- {prediction}")


def print_results(sentences, results):
    print("\n=== جدول التشابه بين كل الأزواج ===")
    print(f"{'الزوج':<8} {'التشابه':>10}  الجملتين")
    print("-" * 90)

    for item in results:
        first = sentences[item["i"] - 1]
        second = sentences[item["j"] - 1]

        print(
            f"{item['i']}↔{item['j']:<5} "
            f"{item['score']:>10.4f}  "
            f"{first} | {second}"
        )


def write_notes(dimensions, first_five, results, bank_score, arabic_results):
    # بنكتب النتائج الفعلية في notes.md عشان التوثيق يبقى ناتج من تشغيل الكود.
    lines = [
        "",
        "---",
        "",
        "# W6D3 — Embeddings",
        "",
        "## الموديل المستخدم",
        "",
        "استخدمت موديل محلي مجاني اسمه "
        "sentence-transformers/all-MiniLM-L6-v2 عن طريق مكتبة "
        "sentence-transformers، فالتجربة نفسها مش محتاجة API Key.",
        "",
        "## يعني إيه Embedding؟",
        "",
        "بالنسبة لي الـEmbedding هو تحويل النص لقائمة أرقام "
        "تمثل المعنى بطريقة يقدر الكمبيوتر يقارنها رياضيًا. "
        "فممكن جملتين يستخدموا كلمات مختلفة لكن يكون معناهم قريب، "
        "وساعتها تمثيلهم الرقمي ممكن يطلع قريب برضه.",
        "",
        "وده مهم للـRAG لأننا بعد كده نقدر نبحث عن النصوص حسب المعنى "
        "مش حسب تطابق الكلمات بس.",
        "",
        "## Semantic vs Keyword",
        "",
        "التشابه بالكلمات مش دايمًا معناه تشابه في المعنى. "
        "وممكن كلمة واحدة تظهر في سياقين مختلفين وتكون بتشير لمعنيين مختلفين. "
        "التجربة هنا بتحاول تبين الفرق ده باستخدام الـEmbeddings.",
        "",
        "## عدد الأبعاد وأول 5 أرقام",
        "",
        f"- عدد الأبعاد: **{dimensions}**",
        f"- أول 5 أرقام من أول Embedding: {[round(x, 6) for x in first_five]}",
        "",
        "## توقعي قبل المقارنة",
        "",
        "### أعلى 3 أزواج متوقعة",
        "",
    ]

    lines.extend(f"- {prediction}" for prediction in PREDICTIONS_TOP_3)

    lines.extend(
        [
            "",
            "### أقل 3 أزواج متوقعة",
            "",
        ]
    )

    lines.extend(f"- {prediction}" for prediction in PREDICTIONS_BOTTOM_3)

    lines.extend(
        [
            "",
            "## النتائج الفعلية",
            "",
            "| الزوج | الجملة الأولى | الجملة الثانية | التشابه |",
            "|---|---|---|---:|",
        ]
    )

    for item in results:
        lines.append(
            f"| {item['i']} ↔ {item['j']} | "
            f"{ENGLISH_SENTENCES[item['i'] - 1]} | "
            f"{ENGLISH_SENTENCES[item['j'] - 1]} | "
            f"{item['score']:.4f} |"
        )

    lines.extend(
        [
            "",
            "### أعلى 3 نتائج فعلية",
            "",
        ]
    )

    for item in results[:3]:
        lines.append(
            f"- {item['i']} ↔ {item['j']}: **{item['score']:.4f}**"
        )

    lines.extend(
        [
            "",
            "### أقل 3 نتائج فعلية",
            "",
        ]
    )

    for item in reversed(results[-3:]):
        lines.append(
            f"- {item['i']} ↔ {item['j']}: **{item['score']:.4f}**"
        )

    lines.extend(
        [
            "",
            "## تجربة كلمة bank",
            "",
            "الجملة الأولى فيها bank بمعنى بنك فلوس، "
            "والتانية فيها bank بمعنى ضفة نهر. "
            "رغم إن الكلمة نفسها موجودة في الاتنين، السياق مختلف، "
            "فالمهم نشوف درجة التشابه اللي الموديل هيدينا.",
            "",
            f"- التشابه بين الجملتين: **{bank_score:.4f}**",
            "",
            "## هل توقعاتي طلعت صح؟ وإيه اللي فاجئني؟",
            "",
            "التوقعات اتكتبت قبل المقارنة. بعد التشغيل بنقارن الترتيب "
            "الفعلي بأعلى وأقل 3 أزواج كنت متوقعهم. "
            "لو ترتيب معين اختلف، ده معناه إن الموديل شايف علاقات دلالية "
            "مش بالضرورة نفس العلاقات اللي أنا توقعتها.",
            "",
            "أكتر حاجة مهمة هنا إن درجة التشابه مش حكم مطلق على معنى الجملتين؛ "
            "هي نتيجة مرتبطة بالموديل نفسه وبالبيانات اللي دخلتها له.",
            "",
            "## Bonus — تجربة العربي",
            "",
            "جربت 4 جمل عربي بنفس الموديل، وشوفت درجات التشابه بينهم. "
            "الهدف إني ألاحظ هل الموديل بيحافظ على العلاقات الدلالية في العربي "
            "ولا الأداء بيكون أضعف من الإنجليزي.",
            "",
            "| الزوج العربي | التشابه |",
            "|---|---:|",
        ]
    )

    for item in arabic_results:
        lines.append(f"| {item['i']} ↔ {item['j']} | {item['score']:.4f} |")

    lines.extend(
        [
            "",
            "## الخلاصة",
            "",
            "الـEmbedding بيحوّل النص لتمثيل رقمي ثابت الطول، "
            "والفكرة الأساسية إن قرب المتجهات ممكن يعكس قرب المعنى. "
            "لكن جودة النتيجة بتعتمد على الموديل واللغة والمجال، "
            "وده سبب إن اختيار موديل الـEmbeddings مهم جدًا قبل بناء RAG.",
            "",
            "## مصادر اليوم",
            "",
            "- 3Blue1Brown — GPT / Word Embeddings: "
            "https://www.3blue1brown.com/lessons/gpt",
            "- Sentence Transformers: "
            "https://pypi.org/project/sentence-transformers/",
            "- Google Gemini Embeddings: "
            "https://ai.google.dev/gemini-api/docs/embeddings?hl=ar",
            "",
        ]
    )

    # بنضيف Section جديد من غير ما نمسح توثيق الأيام اللي فاتت.
    with NOTES_FILE.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    print(f"تحميل الموديل: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("\n=== Embeddings للـ10 جمل الإنجليزية ===")
    english_embeddings = model.encode(
        ENGLISH_SENTENCES,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    dimensions = int(english_embeddings.shape[1])
    first_five = english_embeddings[0][:5].tolist()

    print(f"عدد الأبعاد: {dimensions}")
    print(f"أول 5 أرقام: {[round(x, 6) for x in first_five]}")

    # مهم: التوقعات بتظهر قبل حساب التشابه، زي المطلوب في التاسك.
    print_predictions()

    results = pair_results(ENGLISH_SENTENCES, english_embeddings)
    print_results(ENGLISH_SENTENCES, results)

    print("\n=== أعلى 3 نتائج فعلية ===")
    for item in results[:3]:
        print(f"{item['i']} ↔ {item['j']}: {item['score']:.4f}")

    print("\n=== أقل 3 نتائج فعلية ===")
    for item in reversed(results[-3:]):
        print(f"{item['i']} ↔ {item['j']}: {item['score']:.4f}")

    print("\n=== تجربة bank ===")
    bank_embeddings = model.encode(
        BANK_SENTENCES,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    bank_score = cosine_score(bank_embeddings[0], bank_embeddings[1])
    print(f"تشابه جملتي bank: {bank_score:.4f}")

    print("\n=== Bonus: العربي ===")
    arabic_embeddings = model.encode(
        ARABIC_SENTENCES,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    arabic_results = pair_results(ARABIC_SENTENCES, arabic_embeddings)

    for item in arabic_results:
        print(f"{item['i']} ↔ {item['j']}: {item['score']:.4f}")

    write_notes(
        dimensions,
        first_five,
        results,
        bank_score,
        arabic_results,
    )

    print("\nW6D3 خلص. النتائج الفعلية اتسجلت في notes.md")


if __name__ == "__main__":
    main()

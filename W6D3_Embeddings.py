from pathlib import Path
from itertools import combinations
from datetime import datetime

from sentence_transformers import SentenceTransformer, util


NOTES_FILE = Path("notes.md")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# التثبيت لو المكتبة مش موجودة:
# pip install -U sentence-transformers

ENGLISH_SENTENCES = [
    "The striker scored the winning goal in the final minute.",
    "The forward found the net to secure victory at the end of the match.",
    "The goalkeeper made a brilliant save to stop the penalty.",
    "The goalkeeper blocked the spot kick with an excellent save.",
    "The Python program reads data from a file and processes it.",
    "The script loads information from a file and then transforms it.",
    "The developer fixed a bug in the authentication system.",
    "The programmer repaired an error in the login mechanism.",
    "The chef prepared a spicy chicken dish with fresh vegetables.",
    "The cook made a hot chicken meal using fresh vegetables.",
]

# نفس كلمة bank لكن بمعنيين مختلفين.
BANK_SENTENCES = [
    "I deposited my salary at the bank this morning.",
    "We sat on the river bank and watched the water.",
]

ARABIC_SENTENCES = [
    "سجل المهاجم هدف الفوز في الدقيقة الأخيرة.",
    "أحرز اللاعب الأمامي هدفًا حاسمًا في نهاية المباراة.",
    "المبرمج أصلح خطأ في نظام تسجيل الدخول.",
    "الطاهي حضر وجبة دجاج حارة بالخضروات الطازجة.",
]

PREDICTIONS_TOP_3 = [
    "الجملة 1 مع الجملة 2: نفس فكرة هدف الفوز في نهاية المباراة رغم اختلاف الصياغة.",
    "الجملة 3 مع الجملة 4: نفس معنى تصدي حارس المرمى لركلة الجزاء.",
    "الجملة 5 مع الجملة 6: نفس معنى قراءة البيانات من ملف ثم معالجتها.",
]

PREDICTIONS_BOTTOM_3 = [
    "الجملة 2 مع الجملة 9: كرة القدم مقابل الأكل، فالموضوع مختلف تمامًا.",
    "الجملة 4 مع الجملة 7: كرة القدم مقابل البرمجة، والمفاهيم بعيدة.",
    "الجملة 3 مع الجملة 10: كرة القدم مقابل الأكل، والمفاهيم بعيدة.",
]


def cosine_score(a, b):
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


def append_notes(dimensions, first_five, results, bank_score, arabic_results):
    with NOTES_FILE.open("a", encoding="utf-8") as file:
        file.write("\n\n---\n\n")
        file.write("# W6D3 — Embeddings\n\n")
        file.write(f"وقت التشغيل: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        file.write("## أنا استخدمت إيه؟\n\n")
        file.write(
            "استخدمت موديل محلي مجاني اسمه sentence-transformers/all-MiniLM-L6-v2 "
            "عن طريق مكتبة sentence-transformers، يعني التجربة مش محتاجة API Key. "
            "الموديل بيحوّل كل جملة لتمثيل رقمي ثابت الطول، وبعدها قست التشابه "
            "بين التمثيلات.\n\n"
        )

        file.write("## يعني إيه Embedding؟\n\n")
        file.write(
            "بالنسبة لي الـEmbedding هو طريقة لتحويل الجملة لقائمة أرقام تمثل "
            "معناها في صورة يقدر الكمبيوتر يقارنها رياضيًا. الجمل اللي معناها "
            "قريب ممكن تمثيلاتها تبقى قريبة حتى لو استخدمت كلمات مختلفة. وده "
            "هو الأساس اللي هنحتاجه بعد كده في البحث الدلالي والـRAG.\n\n"
        )

        file.write("## عدد الأبعاد وأول 5 أرقام\n\n")
        file.write(f"- عدد الأبعاد: {dimensions}\n")
        file.write(f"- أول 5 أرقام من أول جملة: {first_five}\n\n")

        file.write("## توقعي قبل المقارنة\n\n")
        file.write("### أعلى 3 أزواج متوقعة\n")
        for prediction in PREDICTIONS_TOP_3:
            file.write(f"- {prediction}\n")

        file.write("\n### أقل 3 أزواج متوقعة\n")
        for prediction in PREDICTIONS_BOTTOM_3:
            file.write(f"- {prediction}\n")

        file.write("\n## نتائج التشابه بين كل الأزواج\n\n")
        file.write("| الزوج | الجملة الأولى | الجملة الثانية | التشابه |\n")
        file.write("|---|---|---|---:|\n")
        for item in results:
            file.write(
                f"| {item['i']} ↔ {item['j']} | "
                f"{ENGLISH_SENTENCES[item['i'] - 1]} | "
                f"{ENGLISH_SENTENCES[item['j'] - 1]} | "
                f"{item['score']:.4f} |\n"
            )

        file.write("\n## أعلى 3 نتائج فعلية\n\n")
        for item in results[:3]:
            file.write(f"- {item['i']} ↔ {item['j']}: {item['score']:.4f}\n")

        file.write("\n## أقل 3 نتائج فعلية\n\n")
        for item in results[-3:][::-1]:
            file.write(f"- {item['i']} ↔ {item['j']}: {item['score']:.4f}\n")

        file.write("\n## جملتا كلمة bank\n\n")
        file.write(
            "الجملة الأولى بتستخدم bank بمعنى بنك فلوس، والتانية بتستخدمها "
            "بمعنى ضفة نهر. سجلت التشابه بينهم عشان أشوف هل الموديل هيميز "
            "المعنى من السياق ولا هيتأثر بالكلمة المشتركة.\n\n"
        )
        file.write(f"- التشابه بين الجملتين: {bank_score:.4f}\n\n")

        file.write("## هل التوقعات طلعت صح؟\n\n")
        file.write(
            "بعد التشغيل أقارن أعلى وأقل 3 نتائج بالتوقعات اللي كتبتها قبل "
            "المقارنة. المهم إني ما أغيرش التوقع بعد ما أشوف الأرقام.\n\n"
        )

        file.write("## تجربة العربي Bonus\n\n")
        file.write(
            "جربت 4 جمل عربي بنفس الموديل، عشان أشوف هل بيحافظ على التشابه "
            "بين الجمل العربية القريبة في المعنى ولا أداؤه أضعف من الإنجليزي.\n\n"
        )
        file.write("| الزوج العربي | التشابه |\n")
        file.write("|---|---:|\n")
        for item in arabic_results:
            file.write(f"| {item['i']} ↔ {item['j']} | {item['score']:.4f} |\n")

        file.write("\n## ملاحظتي النهائية\n\n")
        file.write(
            "الرقم نفسه مش معناه إن الجملة صح أو غلط؛ هو مقياس لقرب التمثيل "
            "الدلالي حسب الموديل اللي اخترته. وده معناه إن اختيار موديل "
            "الـEmbeddings بيفرق، خصوصًا لما اللغة أو المجال يتغير.\n"
        )


def main():
    print(f"تحميل الموديل: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("تحويل الجمل الإنجليزية إلى Embeddings...")
    english_embeddings = model.encode(
        ENGLISH_SENTENCES,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    dimensions = english_embeddings.shape[1]
    first_five = english_embeddings[0][:5].tolist()

    print(f"عدد الأبعاد: {dimensions}")
    print(f"أول 5 أرقام: {first_five}")

    results = pair_results(ENGLISH_SENTENCES, english_embeddings)

    print("\nأعلى 3 أزواج:")
    for item in results[:3]:
        print(f"{item['i']} ↔ {item['j']}: {item['score']:.4f}")

    print("\nأقل 3 أزواج:")
    for item in results[-3:][::-1]:
        print(f"{item['i']} ↔ {item['j']}: {item['score']:.4f}")

    bank_embeddings = model.encode(
        BANK_SENTENCES,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    bank_score = cosine_score(bank_embeddings[0], bank_embeddings[1])

    print(f"\nتشابه جملتي bank: {bank_score:.4f}")

    print("\nتحويل الجمل العربية إلى Embeddings...")
    arabic_embeddings = model.encode(
        ARABIC_SENTENCES,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    arabic_results = pair_results(ARABIC_SENTENCES, arabic_embeddings)

    print("\nنتائج العربي:")
    for item in arabic_results:
        print(f"{item['i']} ↔ {item['j']}: {item['score']:.4f}")

    append_notes(
        dimensions,
        first_five,
        results,
        bank_score,
        arabic_results,
    )

    print("\nW6D3 خلص. النتائج اتضافت لـ notes.md")


if __name__ == "__main__":
    main()

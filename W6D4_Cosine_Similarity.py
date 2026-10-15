from pathlib import Path
from itertools import combinations
from math import sqrt
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer


NOTES_FILE = Path("notes.md")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SENTENCES = [
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

QUERY = "How can a Python program read and process information from a file?"

# الحساب اليدوي المطلوب في التاسك.
A = np.array([1.0, 2.0, 3.0])
B = np.array([2.0, 1.0, 2.0])
C = np.array([-1.0, 2.0, 1.0])


def dot_product(a, b):
    return float(np.sum(a * b))


def magnitude(vector):
    return float(np.sqrt(np.sum(vector * vector)))


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a_magnitude = magnitude(a)
    b_magnitude = magnitude(b)

    if a_magnitude == 0 or b_magnitude == 0:
        return 0.0

    return dot_product(a, b) / (a_magnitude * b_magnitude)


def euclidean_distance(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)))


def rank_pairs(embeddings):
    results = []

    for i, j in combinations(range(len(embeddings)), 2):
        cosine = cosine_similarity(embeddings[i], embeddings[j])
        euclidean = euclidean_distance(embeddings[i], embeddings[j])

        results.append(
            {
                "i": i + 1,
                "j": j + 1,
                "cosine": cosine,
                "euclidean": euclidean,
            }
        )

    return results


def semantic_search(query_embedding, embeddings):
    results = []

    for i, embedding in enumerate(embeddings):
        results.append(
            {
                "index": i + 1,
                "score": cosine_similarity(query_embedding, embedding),
            }
        )

    return sorted(results, key=lambda item: item["score"], reverse=True)


def manual_section():
    dot_ab = dot_product(A, B)
    mag_a = magnitude(A)
    mag_b = magnitude(B)
    cosine_ab = cosine_similarity(A, B)

    dot_ac = dot_product(A, C)
    mag_c = magnitude(C)
    cosine_ac = cosine_similarity(A, C)

    zero_test = cosine_similarity(np.zeros(3), B)

    print("=== الحساب اليدوي ===")
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"C = {C}")
    print(f"A · B = {dot_ab:.6f}")
    print(f"|A| = {mag_a:.6f}")
    print(f"|B| = {mag_b:.6f}")
    print(f"cos(A,B) = {cosine_ab:.6f}")
    print(f"A · C = {dot_ac:.6f}")
    print(f"|C| = {mag_c:.6f}")
    print(f"cos(A,C) = {cosine_ac:.6f}")
    print(f"اختبار Vector صفري = {zero_test:.6f}")

    return {
        "dot_ab": dot_ab,
        "mag_a": mag_a,
        "mag_b": mag_b,
        "cosine_ab": cosine_ab,
        "dot_ac": dot_ac,
        "mag_c": mag_c,
        "cosine_ac": cosine_ac,
        "zero_test": zero_test,
    }


def append_notes(manual, pair_results, search_results, scaled_result):
    with NOTES_FILE.open("a", encoding="utf-8") as file:
        file.write("\n\n---\n\n")
        file.write("# W6D4 — Cosine Similarity و Semantic Search\n\n")
        file.write(f"وقت التشغيل: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        file.write("## الحساب بإيدي\n\n")
        file.write("استخدمت المتجهات الصغيرة دي:\n\n")
        file.write(f"- A = {A.tolist()}\n")
        file.write(f"- B = {B.tolist()}\n")
        file.write(f"- C = {C.tolist()}\n\n")

        file.write("### الزوج A و B\n\n")
        file.write(f"- Dot Product = {manual['dot_ab']:.6f}\n")
        file.write(f"- Magnitude A = {manual['mag_a']:.6f}\n")
        file.write(f"- Magnitude B = {manual['mag_b']:.6f}\n")
        file.write(f"- Cosine Similarity = {manual['cosine_ab']:.6f}\n\n")

        file.write("### الزوج A و C\n\n")
        file.write(f"- Dot Product = {manual['dot_ac']:.6f}\n")
        file.write(f"- Magnitude A = {manual['mag_a']:.6f}\n")
        file.write(f"- Magnitude C = {manual['mag_c']:.6f}\n")
        file.write(f"- Cosine Similarity = {manual['cosine_ac']:.6f}\n\n")

        file.write(
            "اختبار الـVector الصفري رجّع 0 بدل ما يحصل قسمة على صفر، "
            "وده السلوك اللي حطيته في الدالة بنفسي.\n\n"
        )

        file.write("## يعني إيه Cosine Similarity؟\n\n")
        file.write(
            "الـCosine Similarity بيقيس الزاوية بين متجهين، يعني بيسأل "
            "الاتجاهين قريبين من بعض قد إيه. المعادلة هي Dot Product "
            "مقسوم على حاصل ضرب طول المتجهين. عشان الطول بيتقسم عليه "
            "في المعادلة، تكبير المتجه نفسه مش بيغير الاتجاه، وبالتالي "
            "مش بيغير الـCosine. ده مفيد في النصوص لأن اللي يهمني غالبًا "
            "هو اتجاه المعنى في مساحة الـEmbeddings مش حجم المتجه لوحده.\n\n"
        )

        file.write("## مقارنة Cosine و Euclidean\n\n")
        file.write(
            "الـCosine بيركز على الاتجاه، بينما الـEuclidean بيقيس المسافة "
            "الحرفية بين نقطتين، فبيكون حساس لطول المتجه. عشان كده ممكن "
            "متجهين يبقوا في نفس الاتجاه تقريبًا لكن المسافة بينهم كبيرة "
            "لو واحد أطول بكتير.\n\n"
        )

        file.write("## نتائج نفس أزواج إمبارح\n\n")
        file.write("| الزوج | Cosine بتاعتي | Euclidean |\n")
        file.write("|---|---:|---:|\n")
        for item in pair_results:
            file.write(
                f"| {item['i']} ↔ {item['j']} | "
                f"{item['cosine']:.6f} | {item['euclidean']:.6f} |\n"
            )

        file.write("\nالـCosine هنا محسوب بالدالة اللي كتبتها بنفسي، مش بدالة جاهزة.\n\n")

        file.write("## مقارنة الترتيب\n\n")
        cosine_order = sorted(pair_results, key=lambda x: x["cosine"], reverse=True)
        euclidean_order = sorted(pair_results, key=lambda x: x["euclidean"])

        cosine_top = [f"{x['i']}↔{x['j']}" for x in cosine_order[:5]]
        euclidean_top = [f"{x['i']}↔{x['j']}" for x in euclidean_order[:5]]

        file.write(f"- أعلى أزواج حسب Cosine: {', '.join(cosine_top)}\n")
        file.write(f"- أقرب أزواج حسب Euclidean: {', '.join(euclidean_top)}\n")
        file.write(
            "\nالترتيب ممكن يختلف لأن المقياسين بيسألوا سؤالين مختلفين: "
            "واحد عن الاتجاه وواحد عن المسافة الفعلية.\n\n"
        )

        file.write("## تجربة الحجم\n\n")
        file.write(
            "ضربت أحد الـVectors في 10. النتيجة المتوقعة هي إن الـCosine "
            "يفضل زي ما هو لأن الاتجاه لم يتغير، بينما الـEuclidean "
            "يتغير لأن المسافة بين النقطتين اتغيرت.\n\n"
        )
        file.write(f"- Cosine قبل الضرب = {scaled_result['cosine_before']:.6f}\n")
        file.write(f"- Cosine بعد الضرب ×10 = {scaled_result['cosine_after']:.6f}\n")
        file.write(f"- Euclidean قبل الضرب = {scaled_result['euclidean_before']:.6f}\n")
        file.write(f"- Euclidean بعد الضرب ×10 = {scaled_result['euclidean_after']:.6f}\n\n")

        file.write("## Mini Semantic Search\n\n")
        file.write(f"السؤال: {QUERY}\n\n")
        file.write("### أقرب 3 جمل\n\n")
        for item in search_results[:3]:
            file.write(
                f"{item['index']}. {SENTENCES[item['index'] - 1]} "
                f"— Cosine = {item['score']:.6f}\n"
            )

        file.write("\n### هل النتيجة منطقية؟\n\n")
        file.write(
            "أنا هحكم عليها حسب هل الجمل الأقرب فعلًا مرتبطة بالبرمجة "
            "وقراءة/معالجة البيانات من الملفات. لو النتيجة طلعت بعيدة، "
            "السبب المحتمل إن الموديل نفسه له حدود، أو إن صياغة السؤال "
            "مختلفة عن صياغة الجمل، أو إن بياناتنا صغيرة جدًا.\n\n"
        )

        file.write("## مصادر اليوم\n\n")
        file.write(
            "- 3Blue1Brown — Dot products and duality: "
            "https://www.3blue1brown.com/lessons/dot-products/\n"
        )
        file.write(
            "- مثال مبسط لحساب Cosine Similarity: "
            "https://www.omnicalculator.com/math/cosine-similarity\n"
        )
        file.write(
            "- مقارنة Cosine وEuclidean وDot Product: "
            "https://www.semanticnotion.com/blog/cosine-similarity-vs-dot-product-vs-euclidean-distance\n"
        )


def main():
    manual = manual_section()

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        SENTENCES,
        convert_to_tensor=False,
        normalize_embeddings=False,
    )
    embeddings = np.asarray(embeddings, dtype=float)

    # نتأكد إن الحساب اليدوي يطابق الحساب المتوقع من نفس المعادلة.
    numpy_check = cosine_similarity(A, B)
    if not np.isclose(manual["cosine_ab"], numpy_check, atol=1e-10):
        raise AssertionError("الحساب اليدوي والدالة مش متطابقين.")

    results = rank_pairs(embeddings)

    print("\n=== أعلى 5 حسب Cosine ===")
    for item in sorted(results, key=lambda x: x["cosine"], reverse=True)[:5]:
        print(f"{item['i']} ↔ {item['j']}: {item['cosine']:.6f}")

    print("\n=== أقرب 5 حسب Euclidean ===")
    for item in sorted(results, key=lambda x: x["euclidean"])[:5]:
        print(f"{item['i']} ↔ {item['j']}: {item['euclidean']:.6f}")

    # تجربة الحجم: نفس الاتجاه، لكن طول مختلف.
    original = embeddings[0]
    scaled = original * 10

    scaled_result = {
        "cosine_before": cosine_similarity(original, embeddings[1]),
        "cosine_after": cosine_similarity(scaled, embeddings[1]),
        "euclidean_before": euclidean_distance(original, embeddings[1]),
        "euclidean_after": euclidean_distance(scaled, embeddings[1]),
    }

    query_embedding = model.encode(
        [QUERY],
        convert_to_tensor=False,
        normalize_embeddings=False,
    )[0]

    search_results = semantic_search(query_embedding, embeddings)

    print("\n=== Mini Semantic Search ===")
    for item in search_results[:3]:
        print(
            f"{item['index']}. {SENTENCES[item['index'] - 1]} "
            f"-> {item['score']:.6f}"
        )

    append_notes(manual, results, search_results, scaled_result)

    print("\nW6D4 خلص. النتائج اتضافت لـ notes.md")


if __name__ == "__main__":
    main()

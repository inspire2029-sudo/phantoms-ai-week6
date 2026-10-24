from pathlib import Path
import json
import os
import time

# كل الشغل هنا متقسم عشان نقدر نقارن الـPrompt المعيوب بالـPrompt المظبوط.
# الكود بيستخدم OpenAI API لو OPENAI_API_KEY موجود
# ولو مش موجود بيطبع تعليمات التشغيل بدل ما يحط API Key جوه الريبو

REVIEWS = [
    "اشتريت سماعة Anker من أحمد في المعادي، الصوت ممتاز والبطارية بتعدي اليوم كله. التوصيل جه في معاده. أنصح بيها.",
    "موبايل Samsung A54 كاميرته حلوة جداً بس البطارية بتخلص قبل نص اليوم والشاحن بطيء. مش عارف أنصح بيه ولا لأ.",
    "التوصيل اتأخر أسبوع كامل ومحدش رد على الرسايل. الخدمة وحشة جداً.",
    "برافو عليهم، طلبت الشنطة تيجي في يومين فجت بعد تلات أسابيع، وكمان الزرار اتكسر من أول استخدام. خدمة تحفة فعلاً 👏",
    "لابتوب Lenovo IdeaPad اللي استلمته من فرع سيتي ستارز شغال تمام والسعر مناسب، بس مفيش كرتونة أصلية والتغليف كان ضعيف.",
]

# دي نفس الـ5 Prompts المعيوبة اللي اتطلب تشغيلهم زي ما هم من غير تعديل
BAD_PROMPTS = [
    f"قولي رأيك في الريفيو ده وهل هو كويس ولا وحش، ومتطولش، بس اشرح ليه، وطلعلي أي أسماء أو منتجات مذكورة. الريفيو: {REVIEWS[0]}",
    f"قولي رأيك في الريفيو ده وهل هو كويس ولا وحش، ومتطولش، بس اشرح ليه، وطلعلي أي أسماء أو منتجات مذكورة. الريفيو: {REVIEWS[1]}",
    f"قولي رأيك في الريفيو ده وهل هو كويس ولا وحش، ومتطولش، بس اشرح ليه، وطلعلي أي أسماء أو منتجات مذكورة. الريفيو: {REVIEWS[2]}",
    f"قولي رأيك في الريفيو ده وهل هو كويس ولا وحش، ومتطولش، بس اشرح ليه، وطلعلي أي أسماء أو منتجات مذكورة. الريفيو: {REVIEWS[3]}",
    f"قولي رأيك في الريفيو ده وهل هو كويس ولا وحش، ومتطولش، بس اشرح ليه، وطلعلي أي أسماء أو منتجات مذكورة. الريفيو: {REVIEWS[4]}",
]

SYSTEM_PROMPT = """إنت محلل مراجعات منتجات
مهمتك تحلل الريفيو اعتمادا على النص الموجود فقط
ماتخمنش أي معلومة مش موجودة
لو الاسم أو المنتج مش موجود في النص رجع قائمة فاضية
"""

def build_fixed_prompt(review):
    return f"""System
{SYSTEM_PROMPT}

Context
الريفيو موجود بين العلامتين دول
<review>
{review}
</review>

Instruction
حلل الريفيو وحدد هل التقييم إيجابي أو سلبي أو مختلط
اكتب سبب مختصر مبني على النص فقط
استخرج أسماء الأشخاص والشركات والمنتجات المذكورة حرفيا
لو حاجة مش موجودة ماتخمنهاش

Output Format
رجع JSON صالح فقط من غير أي كلام قبله أو بعده بالشكل ده
{{
  "rating": "positive|negative|mixed",
  "reason": "سبب مختصر",
  "names": ["اسم الشخص أو الشركة"],
  "products": ["اسم المنتج"]
}}
"""

def parse_json(text):
    try:
        value = json.loads(text)
        return isinstance(value, dict) and all(
            key in value for key in ["rating", "reason", "names", "products"]
        )
    except (json.JSONDecodeError, TypeError):
        return False

def call_model(prompt, api_key, temperature):
    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("W6D5_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content

def run_experiment(prompts, api_key, temperature):
    results = []
    for prompt_index, prompt in enumerate(prompts, start=1):
        for run_index in range(1, 4):
            output = call_model(prompt, api_key, temperature)
            if output is None:
                results.append({
                    "prompt": prompt_index,
                    "run": run_index,
                    "output": "لم يتم التشغيل لأن مكتبة OpenAI أو API Key مش متاح",
                    "valid_json": False,
                })
            else:
                results.append({
                    "prompt": prompt_index,
                    "run": run_index,
                    "output": output,
                    "valid_json": parse_json(output),
                })
            time.sleep(0.2)
    return results

def print_results(title, results):
    print(f"\n=== {title} ===")
    for item in results:
        print(f"Prompt {item['prompt']} | Run {item['run']} | JSON={item['valid_json']}")
        print(item["output"])
        print()

def write_results_to_file(bad_results, fixed_results):
    output_path = Path("W6D5_results.md")
    lines = [
        "# W6D5 Results",
        "",
        "الملف ده بيتولد بعد التشغيل الفعلي عشان ما نحطش نتائج متخيلة",
        "",
        "## Before",
        "",
        "| Prompt | Run | JSON صح | الرد |",
        "|---:|---:|---|---|",
    ]

    for item in bad_results:
        lines.append(
            f"| {item['prompt']} | {item['run']} | "
            f"{'نعم' if item['valid_json'] else 'لا'} | {item['output'].replace(chr(10), ' ')} |"
        )

    lines += [
        "",
        "## After",
        "",
        "| Prompt | Run | JSON صح | الرد |",
        "|---:|---:|---|---|",
    ]

    for item in fixed_results:
        lines.append(
            f"| {item['prompt']} | {item['run']} | "
            f"{'نعم' if item['valid_json'] else 'لا'} | {item['output'].replace(chr(10), ' ')} |"
        )

    bad_valid = sum(item["valid_json"] for item in bad_results)
    fixed_valid = sum(item["valid_json"] for item in fixed_results)

    lines += [
        "",
        "## Comparison",
        "",
        "| الحالة | JSON مقروء صح من 15 |",
        "|---|---:|",
        f"| قبل الإصلاح | {bad_valid}/15 |",
        f"| بعد الإصلاح | {fixed_valid}/15 |",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    temperature = float(os.getenv("W6D5_TEMPERATURE", "0.7"))

    if not api_key:
        print("حط OPENAI_API_KEY في البيئة قبل التشغيل")
        print("مثال على لينكس أو WSL")
        print("export OPENAI_API_KEY='مفتاحك'")
        return

    print("تشغيل الـ5 Prompts المعيوبة 3 مرات لكل Prompt")
    bad_results = run_experiment(BAD_PROMPTS, api_key, temperature)

    print("تشغيل النسخة المصلحة 3 مرات لكل Prompt")
    fixed_prompts = [build_fixed_prompt(review) for review in REVIEWS]
    fixed_results = run_experiment(fixed_prompts, api_key, temperature)

    print_results("قبل الإصلاح", bad_results)
    print_results("بعد الإصلاح", fixed_results)
    write_results_to_file(bad_results, fixed_results)

    print("النتائج اتسجلت في W6D5_results.md")

if __name__ == "__main__":
    main()

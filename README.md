# phantoms-ai-week6
phantoms-ai-week6


# W6 — LLMs Week Summary

## W6D1

اتعلمنا أساسيات التعامل مع الـLLM والـAPI وإزاي نبعت طلب ونستقبل Response بشكل برمجي
ركزنا على إعدادات زي الـTemperature وتأثيرها على طبيعة الرد

## W6D2

اشتغلنا على استدعاء الـLLM وفهمنا إن نفس الـPrompt ممكن يطلع ردود مختلفة حسب الإعدادات والسياق
واتعلمنا نتحكم في السلوك بدل ما نعتمد على رد واحد

## W6D3

حوّلنا الجمل لـEmbeddings وشوفنا إزاي المعنى بيتحول لـVector ونقيس قرب الجمل من بعض
واتجربنا Semantic Similarity وكمان حالة كلمة bank بمعنيين مختلفين

## W6D4

فككنا Cosine Similarity بنفسنا وحسبنا Dot Product وMagnitude وEuclidean Distance
وبنينا Mini Semantic Search يرتب الجمل حسب قربها من سؤال جديد

## W6D5

شخصنا Prompt غامض وقارنّا 15 محاولة قبل الإصلاح و15 محاولة بعده
وبنينا Prompt منظم فيه System وContext وInstruction وOutput Format مع Unknown Handling

## أهم حاجة اتعلمتها الأسبوع ده

أهم حاجة إن التعامل مع الـLLM مش مجرد إننا نكتب سؤال حلو ونستنى نتيجة كويسة
لازم نحدد المطلوب ونقيس النتيجة ونكرر التجربة ونخلي الـOutput قابل للمعالجة بالكود

## حاجة لسه مش واضحة عندي

لسه محتاجة أشوف تأثير تغيير الموديل نفسه مع نفس الـPrompt ونفس الـEvaluation عشان أفرق عمليًا بين مشكلة في الـPrompt ومشكلة في قدرات الموديل

## ملفات الأسبوع

- W6D3_Embeddings.py
- W6D4_Cosine_Similarity.py
- W6D5_Prompt_Debugging.py
- notes.md
- W6D5_results.md بعد التشغيل الفعلي

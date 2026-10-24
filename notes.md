# W6D1 — Meet the LLM

## 1. What is a token?

A token is a small piece of text that the language model processes. A token is not always the same as one complete word: a word can be one token, several tokens, or sometimes part of a word can be a token.

Tokens matter because the model does not receive text as an unlimited stream of characters. The input and output are processed within a context window, and token usage can affect limits, speed, and API cost.

## 2. Why is the model described as predicting instead of understanding?

A language model generates text by repeatedly predicting what token is likely to come next based on the tokens that came before it and the patterns learned during training.

Calling this prediction does not mean the model is useless or that its output is random. The predictions can encode a large amount of useful structure and knowledge. However, next-token prediction is different from having human-style understanding, intentions, experiences, or guaranteed factual reasoning.

## 3. Roles: system, user, assistant

The conversation is structured into roles.

- **system**: defines high-level behavior and instructions.
- **user**: contains the user's request.
- **assistant**: contains the model's generated response.

In this task, the same question is sent twice while changing only the system instruction. This demonstrates that the system role can strongly influence the style and level of detail of the response.

## 4. Does the API remember previous requests?

A basic API request should be treated as stateless. The application is responsible for sending the relevant conversation history again when previous context is needed.

That means a chat application can appear to have memory, while the application itself is actually managing and resending the conversation context.

## 5. Example of an incorrect or weak model answer

A useful example is asking a model a question that requires exact counting, such as:

> How many letters are in the word "mississippi"?

A model can sometimes give an incorrect count even though the word is simple. The reason is that the model is generating an answer from learned patterns rather than using a guaranteed symbolic counting procedure every time. Tokenization also means the model does not necessarily process each letter as a separate unit.

This is a good reminder that fluent language is not the same thing as guaranteed correctness.

## 6. What this practical task demonstrates

This exercise connects the concepts to a real API call:

1. The API key is stored outside the source code.
2. Python sends a system message and a user message.
3. The model returns an assistant message.
4. The same question is tested with two different system prompts.
5. Token usage is read from the API response instead of being guessed.

The API key must never be committed to GitHub.


---

# W6D2 — Temperature و Hallucination

## الفكرة ببساطة

النهارده الفكرة الأساسية عندي كانت إني أشوف بعيني إن نفس السؤال مش لازم يطلع منه نفس الرد كل مرة، وإن الموديل ممكن يقول حاجة غلط وهو واثق جداً منها.

### الـTemperature

الـTemperature بتأثر على طريقة اختيار الموديل للتوكن اللي بعده.

- **0.1**: الردود المفروض تبقى أهدى وأثبت وأقل تنوع.
- **0.9**: مساحة التنوع أكبر، فممكن الصياغة تختلف أكتر من مرة للتانية.

المهم إن Temperature مش زرار "صح وغلط". لما أعليها أنا بزوّد التنوع، وده ممكن يزوّد الاختلاف وأحياناً يزوّد فرصة إن الرد يطلع أقل دقة.

## تجربة الـTemperature

استخدمت نفس النص ونفس طلب التلخيص 3 مرات عند 0.1 و3 مرات عند 0.9.

| الحرارة | التجربة | النتيجة |
|---:|---:|---|
| 0.1 | 1 | النتيجة بتتسجل تلقائياً بعد تشغيل الملف |
| 0.1 | 2 | النتيجة بتتسجل تلقائياً بعد تشغيل الملف |
| 0.1 | 3 | النتيجة بتتسجل تلقائياً بعد تشغيل الملف |
| 0.9 | 1 | النتيجة بتتسجل تلقائياً بعد تشغيل الملف |
| 0.9 | 2 | النتيجة بتتسجل تلقائياً بعد تشغيل الملف |
| 0.9 | 3 | النتيجة بتتسجل تلقائياً بعد تشغيل الملف |

الملف W6D2_Temperature_Hallucination.py هو اللي بيشغل الست تجارب وبيضيف الردود الفعلية للجدول تلقائياً.

## الـHallucination

الـHallucination بالنسبة لي هنا مش مجرد إن الموديل غلط. المشكلة الأكبر إنه ممكن يغلط **بشكل مقنع جداً**، فيخلي اللي قدامه يفتكر إن الكلام متأكد منه.

جربت 5 أنواع أسئلة:

1. سؤال معلوماته سهلة التحقق: عاصمة أستراليا.
2. سؤال معلوماته سهلة التحقق: مؤلف رواية Pride and Prejudice.
3. سؤال عن كتاب مخترع.
4. سؤال عن مكتبة Python مخترعة.
5. سؤال عن حدث حديث جداً، وطلبت من الموديل يقول إنه مش متأكد لو معندوش معلومة موثوقة.

| النوع | السؤال | هل هلوس؟ | إزاي أتحقق؟ |
|---|---|---|---|
| معلومة حقيقية | عاصمة أستراليا | يتحدد بعد تشغيل النموذج | أراجع مصدر موثوق |
| معلومة حقيقية | مؤلف Pride and Prejudice | يتحدد بعد تشغيل النموذج | أراجع مصدر موثوق |
| حاجة مخترعة | The Glass Orchard of Novera | يتحدد بعد تشغيل النموذج | أدور على وجود الكتاب ومصدره |
| حاجة مخترعة | quantumpicklex | يتحدد بعد تشغيل النموذج | أتحقق من وجود مكتبة حقيقية ومصدر رسمي |
| حدث حديث جداً | حدث اليوم | يتحدد بعد تشغيل النموذج | أقارن بتاريخ ومصدر أخبار موثوق |

ولو الموديل قال "معرفش" أو قال إن الحاجة مش موجودة، أنا هعتبر دي **نتيجة كويسة** في التجربة، مش فشل. لأن الاعتراف بعدم المعرفة أحسن من إنه يخترع إجابة.

## تجربة تقليل الهلوسة

أخدت سؤال الكتاب المخترع وجربته بطريقتين.

**قبل تعديل الـPrompt:**  
طلبت من الموديل يجاوب بشكل مباشر.

**بعد تعديل الـPrompt:**  
وضحت له إن الكتاب المذكور مش معروف في السياق، وإنه لو مش قادر يتحقق من وجوده أو من مؤلفه يقول "مش متأكد" بدل ما يخترع اسم.

الهدف هنا مش إني أثبت إن Prompt واحد هيحل المشكلة. الهدف أشوف هل إعطاء الموديل مساحة واضحة إنه يعترف بعدم اليقين يقلل التخمين ولا لأ.

## هل نقدر نلغي الهلوسة نهائياً؟

لا، مش شايف إننا نقدر نلغيها نهائياً في نظام حقيقي. نقدر نقللها ونكشفها، لكن مينفعش أبني النظام على افتراض إن الموديل عمره ما هيغلط.

لو هبني نظام حقيقي، هعتمد على أكتر من طبقة:

- لما السؤال محتاج حقيقة محددة، أجيبها من مصدر موثوق بدل ما أسيب الموديل يخمن.
- أخلي الموديل يقول إنه مش متأكد لما مفيش معلومات كفاية.
- أعمل تحقق مستقل للحاجات المهمة.
- في المعلومات الحديثة، أعتمد على مصدر خارجي محدث بدل معرفة الموديل لوحدها.
- ماعتبرش إن طريقة كلام الموديل الواثقة دليل إن المعلومة صحيحة.

الفكرة في الآخر إن **تقليل الهلوسة مش معناه إننا نثق في الموديل أكتر، لكن إننا نصمم النظام بحيث غلط الموديل مايتحولش لمشكلة.**

## مصادر اليوم

- بحث OpenAI عن سبب هلوسة النماذج: https://openai.com/index/why-language-models-hallucinate/
- فيديو 3Blue1Brown عن الـTransformers، وفيه شرح للـTemperature عند جزء الـSoftmax: https://www.youtube.com/watch?v=wjZofJX0v4M


---

# W6D3 — Embeddings

## الموديل المستخدم

استخدمت موديل محلي مجاني اسمه sentence-transformers/all-MiniLM-L6-v2 عن طريق مكتبة sentence-transformers، فالتجربة نفسها مش محتاجة API Key.

## يعني إيه Embedding؟

بالنسبة لي الـEmbedding هو تحويل النص لقائمة أرقام تمثل المعنى بطريقة يقدر الكمبيوتر يقارنها رياضيًا. فممكن جملتين يستخدموا كلمات مختلفة لكن يكون معناهم قريب، وساعتها تمثيلهم الرقمي ممكن يطلع قريب برضه.

وده مهم للـRAG لأننا بعد كده نقدر نبحث عن النصوص حسب المعنى مش حسب تطابق الكلمات بس.

## Semantic vs Keyword

التشابه بالكلمات مش دايمًا معناه تشابه في المعنى. وممكن كلمة واحدة تظهر في سياقين مختلفين وتكون بتشير لمعنيين مختلفين. التجربة هنا بتحاول تبين الفرق ده باستخدام الـEmbeddings.

## عدد الأبعاد وأول 5 أرقام

الملف W6D3_Embeddings.py بيطبع عدد الأبعاد وأول 5 أرقام من أول Embedding بعد تشغيل الموديل.

## توقعي قبل المقارنة

### أعلى 3 أزواج متوقعة

- الجملة 1 مع الجملة 2: الاتنين بيتكلموا عن تسجيل هدف متأخر أدى للفوز.
- الجملة 3 مع الجملة 4: الاتنين عن حارس مرمى منع تسجيل هدف من ركلة جزاء.
- الجملة 5 مع الجملة 6: الاتنين عن قراءة بيانات من ملف ومعالجتها.

### أقل 3 أزواج متوقعة

- الجملة 2 مع الجملة 9: كورة مقابل أكل، فالمعنى بعيد.
- الجملة 4 مع الجملة 7: كورة مقابل برمجة، فالمعنى بعيد.
- الجملة 3 مع الجملة 10: كورة مقابل أكل، فالمعنى بعيد.

## النتائج

الملف بيحسب التشابه بين كل الأزواج باستخدام الدالة الجاهزة في sentence-transformers، وبيطبع جدول التشابه وأعلى 3 وأقل 3 نتائج فعلية.

وبعد التشغيل بيضيف نفس النتائج هنا تلقائيًا، مع عدد الأبعاد وأول 5 أرقام ونتيجة تجربة bank ونتائج العربي.

## تجربة كلمة bank

استخدمت كلمة bank في جملتين بمعنيين مختلفين: بنك فلوس وضفة نهر.

الهدف أشوف هل الموديل هيتأثر بالكلمة المشتركة ولا هيقدر يميز المعنى من السياق.

الملف بيحسب درجة التشابه بين الجملتين بشكل منفصل ويسجلها في notes.md.

## هل توقعاتي طلعت صح وإيه اللي فاجئني؟

التوقعات اتكتبت قبل المقارنة عشان ما أغيرهاش بعد ما أشوف الأرقام.

بعد التشغيل أقارن أعلى وأقل 3 نتائج فعلية بالتوقعات. لو ترتيب معين اختلف، ده معناه إن الموديل شايف علاقة دلالية مختلفة عن توقعي.

أهم حاجة بالنسبة لي إن درجة التشابه مش حكم مطلق على معنى الجملتين؛ هي نتيجة مرتبطة بالموديل نفسه وباللغة والمجال والجمل اللي استخدمتها.

## Bonus — تجربة العربي

جربت 4 جمل عربي بنفس الموديل عشان أشوف هل بيحافظ على العلاقات الدلالية في العربي ولا الأداء بيكون أضعف من الإنجليزي.

الملف بيحسب كل الأزواج العربية ويسجل النتائج في جدول تلقائيًا.

## الخلاصة

الـEmbedding بيحوّل النص لتمثيل رقمي ثابت الطول، والفكرة الأساسية إن قرب المتجهات ممكن يعكس قرب المعنى.

وده الأساس اللي هنحتاجه في البحث الدلالي والـRAG، مع ملاحظة إن جودة النتيجة بتعتمد على الموديل واللغة والمجال.

## مصادر اليوم

- 3Blue1Brown — GPT / Word Embeddings: https://www.3blue1brown.com/lessons/gpt
- Sentence Transformers: https://pypi.org/project/sentence-transformers/
- Google Gemini Embeddings: https://ai.google.dev/gemini-api/docs/embeddings?hl=ar

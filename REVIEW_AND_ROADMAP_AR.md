# تقييم النواة — Arabic Seed

## الحكم

**نعم، يمكن اعتبار المشروع نواة تعليمية قابلة للتوسع، لكن لا يصح اعتباره بعد نموذجًا لغويًا عربيًا قابلًا للتربية.**

المشروع الحالي يحقق أهم شيء في البداية: نموذج Transformer decoder-only يبدأ بأوزان عشوائية، يحول النص إلى رموز، ويتعلم توقع الرمز التالي باستخدام cross-entropy.

لكن التشبيه بالطفل يجب استخدامه بحذر:
- الطفل يتعلم من بيئة متعددة الحواس وبآليات تعلم مستمرة وذاكرة وتجربة.
- هذا المشروع يتعلم حاليًا مهمة واحدة: **next-token prediction** من corpus ثابت.
- النمو لا يحدث تلقائيًا بإضافة ملفات؛ يجب تصميم دورة بيانات/تدريب/تقييم/إصدارات.

## نقاط القوة

1. **القلب المعماري صحيح**: Embedding + causal self-attention + MLP + residual + LayerNorm + LM head.
2. **الوزن الحقيقي يبدأ من الصفر**: لا توجد أوزان نموذج جاهز.
3. **المسار مفهوم وقابل للتعليم**: tokenizer → dataset → Transformer → loss → optimizer → checkpoint → generation.
4. **الفصل جيد نسبيًا** بين tokenizer/model/training/generation.
5. **الـ vocab مرن**؛ الانتقال إلى subword tokenizer ممكن معماريًا.
6. **checkpoint يحتوي config**، ما يجعل إعادة بناء المعمارية ممكنة.
7. النسخة المحسنة تضيف **UNK، تحققًا من حجم البيانات، gradient clipping، scheduler، best/last checkpoints، perplexity، واختبارات pipeline**.

## نقاط الضعف الحرجة

### 1. corpus الحالي صغير للغاية

البيانات حوالي 1.1 ألف حرف فقط. هذا لا يكفي لتعلم العربية؛ سيؤدي غالبًا إلى الحفظ والتقليد السطحي.

### 2. خلل مسار البيانات

النسخة الأصلية تحتوي `corpus.txt` في جذر المشروع، بينما `train.py` يبحث عن `data/corpus.txt`. لذلك لن يبدأ التدريب كما هو.

### 3. character tokenizer

ممتاز للتعلم، لكنه غير مناسب للتوسع. عدد خطوات التدريب يصبح ضخمًا لأن كل حرف token.

**المرحلة التالية:** SentencePiece/BPE عربي بعد بناء corpus أكبر.

### 4. لا توجد إدارة بيانات حقيقية

ينقص المشروع:
- manifest لمصادر البيانات.
- provenance/licensing.
- normalization.
- deduplication.
- quality filtering.
- document boundaries.
- إصدارات للـ corpus.

وهذه أهم من زيادة عدد الطبقات في المراحل المبكرة.

### 5. لا توجد دورة تقييم حقيقية

loss وحده لا يكفي. نحتاج لاحقًا إلى:
- validation corpus مستقل.
- perplexity.
- اختبارات لغوية عربية.
- اختبارات حفظ/تلوث data contamination.
- قياس جودة التوليد عبر إصدارات النموذج.

### 6. لا توجد آلية استمرار تدريب ناضجة

يمكن مواصلة التدريب، لكن يلزم حفظ:
- model weights
- optimizer state
- scheduler state
- global step
- tokenizer version
- dataset version
- config

حتى نعرف بالضبط ماذا حدث في كل مرحلة.

### 7. لا توجد استراتيجية للتعلم المرحلي

إذا أضفت لاحقًا كمية كبيرة من النصوص، لا يكفي تشغيل التدريب عشوائيًا. نحتاج curriculum/data mixture، مع الاحتفاظ بعينات من البيانات القديمة لتقليل catastrophic forgetting.

## ما الذي يعنيه "بذر الشجرة"؟

النواة الصحيحة ليست أن نجعل النموذج كبيرًا الآن، بل أن نجعل **النظام قابلًا للنمو دون إعادة بنائه**.

التسلسل المقترح:

```text
                 ┌────────────────────┐
                 │   Raw Arabic Data  │
                 └─────────┬──────────┘
                           ↓
                 normalize / filter
                           ↓
                 deduplicate + license
                           ↓
                 versioned corpus
                           ↓
                    tokenizer
                           ↓
                 tokenized dataset
                           ↓
                 ┌─────────────────┐
                 │ GPT Pretraining │
                 └────────┬────────┘
                          ↓
                checkpoint + metadata
                          ↓
              evaluation / regression
                          ↓
                 next training cycle
```

## المراحل المقترحة

### Seed-0 — الآن

الهدف: فهم وتشغيل السلسلة.

- character tokenizer
- 4 layers / 128 embedding
- corpus صغير
- CPU
- اختبارات pipeline

### Seed-1 — تأسيس اللغة

لا تكبر النموذج كثيرًا بعد.

أولًا ابنِ corpus عربيًا نظيفًا من مصادر مرخصة، ثم:
- normalization مضبوط للعربية
- document boundaries
- dedup
- provenance
- train/validation/test split
- SentencePiece/BPE

### Seed-2 — النمو الحقيقي

ابدأ بتوسيع:
- عدد tokens
- context length
- model width/depth
- batch/effective batch
- GPU training
- checkpoint resume
- experiment tracking

### Seed-3 — المعرفة والسلوك

بعد pretraining حقيقي:
- continued pretraining على مجالات مختارة.
- SFT للتعليمات.
- preference/alignment لاحقًا.
- اختبارات عربية متخصصة.

## نقطة مهمة جدًا

**لا أنصح بإضافة "ذاكرة" أو قاعدة معرفة إلى النموذج الآن وتسميتها تعلمًا.**

إذا كان الهدف نموذجًا يتعلم اللغة من الصفر، فالمسار الأساسي يجب أن يبقى:

`بيانات → tokens → prediction → loss → gradient → weights`

أما الذاكرة الخارجية/RAG/قاعدة المعرفة فهي طبقة مختلفة، وليست بديلًا عن تعلم النموذج.

## ما أضفته في هذه النسخة

- إصلاح مسار corpus.
- إضافة `data/corpus.txt`.
- tokenizer أكثر أمانًا مع `<unk>`.
- CLI لإعدادات التدريب.
- validation واضح.
- perplexity.
- gradient clipping.
- cosine learning-rate schedule.
- `best.pt` و`last.pt`.
- اختبارات pipeline.
- توثيق خطة النمو.
- تحذير صريح من اعتبار corpus الحالي بيانات تدريب حقيقية.

## ما لم أفعله عمدًا

لم أحول المشروع مباشرة إلى Llama أو أضف BPE أو RoPE أو FlashAttention.

السبب: **النواة التعليمية الحالية مفيدة لفهم ما يحدث.**

الأفضل أن نحافظ على Seed-0 كمرجع، ثم نبني Seed-1 فوقه بطريقة يمكن قياسها. لا نريد أن يصبح المشروع صندوقًا أسود لمجرد أنه أصبح أكبر.

## الخلاصة

يمكن أن تكون هذه النواة "البذرة"، بشرط أن يكون التطوير القادم متمحورًا حول:

**البيانات أولًا → tokenizer → pipeline → القياس → ثم زيادة حجم النموذج.**

ليس:

**زيادة النموذج → أمل أن يتعلم.**

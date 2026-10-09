# Arabic Seed — نواة نموذج لغوي عربي

هذه نسخة محسنة من النموذج التجريبي الأصلي.

## تشغيل

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest -q
python3 model_baseline.py
python3 train.py
python3 generate.py --prompt "السلام عليكم"
```

> `data/corpus.txt` الحالي مجرد corpus اختبار صغير جدًا. لا تتوقع منه لغة عربية حقيقية.

## مبدأ المشروع

الهدف هو بناء نموذج لغوي من الصفر مع قابلية التوسع التدريجي، مع إبقاء:

**البيانات → tokenizer → model → training → checkpoint → evaluation**

واضحة وقابلة للتدقيق.

راجع `REVIEW_AND_ROADMAP_AR.md` قبل زيادة حجم النموذج.

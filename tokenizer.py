"""
Arabic Seed Tokenizer
مرحلة أولى تعليمية: character-level مع رمز UNK.
لا نستبدله بـ BPE إلا بعد تجهيز corpus حقيقي واختبارات tokenization.
"""
import json

class CharTokenizer:
    UNK = "<unk>"

    def __init__(self, vocab=None):
        self.vocab = vocab or [self.UNK]
        self.stoi = {s:i for i,s in enumerate(self.vocab)}
        self.itos = {i:s for i,s in enumerate(self.vocab)}

    @classmethod
    def build_from_text(cls, text):
        chars = sorted(set(text))
        vocab = [cls.UNK] + [c for c in chars if c != cls.UNK]
        return cls(vocab)

    @property
    def vocab_size(self): return len(self.vocab)

    def encode(self, text):
        unk = self.stoi[self.UNK]
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, tokens):
        return "".join(self.itos[int(t)] for t in tokens if int(t) != self.stoi[self.UNK])

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type":"char","vocab":self.vocab}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            obj=json.load(f)
        return cls(obj["vocab"])

"""
model.py
---------
معمارية Transformer من نوع GPT (decoder-only) مبنية من الصفر،
بدون استخدام مكتبة نماذج جاهزة (لا transformers ولا غيرها) —
فقط PyTorch الأساسي (طبقات، مصفوفات، دوال تفعيل).

هذا هو "القلب" المعماري الذي سيكبر معك لاحقًا:
نفس البنية بالضبط تُستخدم في النماذج الكبيرة، والفرق فقط هو
عدد الطبقات (n_layer) وحجم التمثيل (n_embd) وعدد الرؤوس (n_head)
وطول السياق (block_size).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int        # عدد الرموز الفريدة التي يعرفها النموذج
    block_size: int = 128  # أقصى طول سياق (context window) بالرموز
    n_layer: int = 4       # عدد طبقات الـ Transformer
    n_head: int = 4        # عدد رؤوس الانتباه (attention heads) في كل طبقة
    n_embd: int = 128      # أبعاد تمثيل كل رمز (embedding dimension)
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """
    طبقة الانتباه الذاتي السببي (causal self-attention):
    كل رمز "ينظر" فقط إلى الرموز التي قبله (وليس بعده) — هذا ما يجعل
    النموذج قادرًا على توليد نص تسلسليًا، رمزًا بعد رمز.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # إسقاط واحد يولّد الاستعلام (Q) والمفتاح (K) والقيمة (V) معًا لكفاءة أعلى
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # قناع سببي: يمنع كل رمز من رؤية ما بعده
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape  # (دفعة, طول السياق, أبعاد التمثيل)

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        # إعادة تشكيل لتوزيع الأبعاد على الرؤوس المتعددة
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # درجات الانتباه = مدى "تشابه" كل استعلام مع كل مفتاح
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class MLP(nn.Module):
    """شبكة تغذية أمامية بسيطة تُطبّق على كل رمز على حدة بعد الانتباه."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """كتلة Transformer واحدة = انتباه + MLP، مع LayerNorm واتصالات متبقّية (residual)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, "طول السياق أكبر من block_size المسموح"

        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        """توليد نص رمزًا رمزًا انطلاقًا من idx (سياق بداية)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


if __name__ == "__main__":
    # اختبار سريع: هل تمر البيانات عبر النموذج دون أخطاء؟
    cfg = GPTConfig(vocab_size=100, block_size=64, n_layer=4, n_head=4, n_embd=128)
    model = GPT(cfg)
    print(f"عدد المعاملات (parameters): {model.num_params():,}")

    dummy_input = torch.randint(0, cfg.vocab_size, (2, 32))
    dummy_target = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(dummy_input, dummy_target)
    print("شكل المخرجات:", logits.shape)
    print("الخسارة الابتدائية:", loss.item())
    print("نجح الاختبار ✓")

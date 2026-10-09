import os, tempfile, torch
from tokenizer import CharTokenizer
from model_baseline import GPT, GPTConfig

def test_roundtrip():
    t=CharTokenizer.build_from_text("سلام عالم")
    assert t.decode(t.encode("سلام عالم"))=="سلام عالم"

def test_model_forward():
    c=GPTConfig(vocab_size=50,block_size=16,n_layer=2,n_head=2,n_embd=32)
    m=GPT(c); x=torch.randint(0,50,(2,16)); y=torch.randint(0,50,(2,16))
    logits,loss=m(x,y)
    assert logits.shape==(2,16,50)
    assert torch.isfinite(loss)

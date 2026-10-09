"""
Train the seed GPT from scratch.

Important:
- The filesystem corpus is the source of truth.
- Run from this project directory.
- This is pretraining (next-token prediction), not instruction tuning.
"""
import argparse, math, os, random, time
import torch
from model_baseline import GPT, GPTConfig
from tokenizer import CharTokenizer

def get_batch(data, block_size, batch_size, device):
    if len(data) <= block_size + 1:
        raise ValueError(f"Corpus split has {len(data)} tokens; need > {block_size+1}.")
    ix = torch.randint(0, len(data)-block_size-1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate(model, train_data, val_data, args, device):
    model.eval(); out={}
    for name,data in (("train",train_data),("val",val_data)):
        vals=[]
        for _ in range(args.eval_iters):
            x,y=get_batch(data,args.block_size,args.batch_size,device)
            _,loss=model(x,y); vals.append(loss.item())
        out[name]=sum(vals)/len(vals)
    model.train(); return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data/corpus.txt")
    ap.add_argument("--steps",type=int,default=2000)
    ap.add_argument("--block-size",type=int,default=128)
    ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--layers",type=int,default=4)
    ap.add_argument("--heads",type=int,default=4)
    ap.add_argument("--embd",type=int,default=128)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--dropout",type=float,default=.1)
    ap.add_argument("--val-ratio",type=float,default=.1)
    ap.add_argument("--eval-interval",type=int,default=200)
    ap.add_argument("--eval-iters",type=int,default=20)
    ap.add_argument("--seed",type=int,default=1337)
    ap.add_argument("--checkpoint-dir",default="checkpoints")
    args=ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.data): raise FileNotFoundError(args.data)
    text=open(args.data,encoding="utf8").read()
    if not text.strip(): raise ValueError("Corpus is empty.")

    tok=CharTokenizer.build_from_text(text)
    ids=torch.tensor(tok.encode(text),dtype=torch.long)
    split=int((1-args.val_ratio)*len(ids))
    train_data,val_data=ids[:split],ids[split:]
    if len(train_data)<=args.block_size+1 or len(val_data)<=args.block_size+1:
        raise ValueError("Corpus is too small for the selected block size and validation split. Add real data or reduce --block-size.")

    cfg=GPTConfig(vocab_size=tok.vocab_size,block_size=args.block_size,
                  n_layer=args.layers,n_head=args.heads,n_embd=args.embd,dropout=args.dropout)
    model=GPT(cfg).to(device)
    print(f"device={device} | chars={len(text):,} | vocab={tok.vocab_size} | params={model.num_params():,}")

    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.1)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps)
    os.makedirs(args.checkpoint_dir,exist_ok=True)
    tok.save(os.path.join(args.checkpoint_dir,"vocab.json"))

    best=float("inf"); start=time.time()
    for step in range(1,args.steps+1):
        x,y=get_batch(train_data,args.block_size,args.batch_size,device)
        _,loss=model(x,y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); scheduler.step()

        if step % args.eval_interval==0 or step==args.steps:
            m=estimate(model,train_data,val_data,args,device)
            ppl=math.exp(min(m["val"],20))
            print(f"step={step:5d} train_loss={m['train']:.4f} val_loss={m['val']:.4f} val_ppl={ppl:.2f} elapsed={time.time()-start:.1f}s")
            state={"model_state_dict":model.state_dict(),"config":cfg,
                   "step":step,"seed":args.seed,"data_path":args.data,
                   "train_chars":len(train_data),"val_chars":len(val_data),
                   "val_loss":m["val"]}
            torch.save(state,os.path.join(args.checkpoint_dir,"last.pt"))
            if m["val"] < best:
                best=m["val"]; torch.save(state,os.path.join(args.checkpoint_dir,"best.pt"))
    print("Training complete.")

if __name__=="__main__": main()

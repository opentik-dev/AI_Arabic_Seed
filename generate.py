import argparse, os, torch
from model_baseline import GPT
from tokenizer import CharTokenizer

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",default="checkpoints/best.pt")
    ap.add_argument("--prompt",default="السلام عليكم")
    ap.add_argument("--max-new-tokens",type=int,default=200)
    ap.add_argument("--temperature",type=float,default=.8)
    ap.add_argument("--top-k",type=int,default=20)
    args=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    d=torch.load(args.checkpoint,map_location=device,weights_only=False)
    tok=CharTokenizer.load(os.path.join(os.path.dirname(args.checkpoint),"vocab.json"))
    model=GPT(d["config"]).to(device); model.load_state_dict(d["model_state_dict"]); model.eval()
    ctx=torch.tensor([tok.encode(args.prompt)],dtype=torch.long,device=device)
    out=model.generate(ctx,args.max_new_tokens,args.temperature,args.top_k)
    print(tok.decode(out[0].tolist()))

if __name__=="__main__": main()

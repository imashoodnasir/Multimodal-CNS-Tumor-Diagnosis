
import argparse, yaml, os, torch, numpy as np
from transformers import AutoTokenizer
from avlt.models.avlt import AVLT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/avlt_synthetic.pt")
    parser.add_argument("--image", type=str, required=False, help="Path to 4-channel numpy .npy (C,H,W)")
    parser.add_argument("--text", type=str, default="Patient with IDH mutation and MGMT methylation.")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AVLT(num_classes=cfg["num_classes"], image_size=cfg["image_size"],
                 backbone=cfg["vision"]["backbone"], text_model=cfg["text"]["model_name"],
                 dropout=cfg["dropout"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    # Prepare image
    if args.image and os.path.exists(args.image):
        arr = np.load(args.image)  # (4,H,W)
    else:
        arr = np.random.randn(4, cfg["image_size"], cfg["image_size"]).astype(np.float32)
    img = torch.tensor(arr).unsqueeze(0).to(device)
    # Prepare text
    tok = AutoTokenizer.from_pretrained(cfg["text"]["model_name"])
    enc = tok(args.text, padding='max_length', truncation=True, max_length=cfg["text_maxlen"], return_tensors='pt')
    ids = enc["input_ids"].to(device); attn = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits, f_v, f_t, f_fused, a, b = model(img, ids, attn)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    print("Probabilities:", prob)

if __name__ == "__main__":
    main()

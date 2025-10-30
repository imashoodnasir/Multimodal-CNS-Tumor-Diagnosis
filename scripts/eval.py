
import argparse, yaml, os, json, torch
from avlt.train.engine import create_dataloaders
from avlt.models.avlt import AVLT
from avlt.utils.metrics import MetricTracker
from avlt.viz.plots import save_confusion, save_roc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/avlt_synthetic.pt")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, val_dl = create_dataloaders(cfg)
    model = AVLT(num_classes=cfg["num_classes"], image_size=cfg["image_size"],
                 backbone=cfg["vision"]["backbone"], text_model=cfg["text"]["model_name"],
                 dropout=cfg["dropout"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    metrics = MetricTracker(cfg["num_classes"])
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in val_dl:
            imgs = batch["image"].to(device)
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["label"].to(device)
            logits, *_ = model(imgs, ids, attn)
            metrics.update(logits, y)
            y_true.append(y.cpu())
            y_prob.append(torch.softmax(logits, dim=1).cpu())
    y_true = torch.cat(y_true).numpy()
    y_prob = torch.cat(y_prob).numpy()
    report = metrics.report()
    print("Eval metrics:", report)
    os.makedirs(cfg["outputs"], exist_ok=True)
    save_confusion(y_true, y_prob.argmax(1), os.path.join(cfg["outputs"], "confusion_eval.png"))
    save_roc(y_true, y_prob, os.path.join(cfg["outputs"], "roc_eval.png"))
    with open(os.path.join(cfg["outputs"], "metrics_eval.json"), "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()

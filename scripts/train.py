
import argparse, yaml, os
from avlt.train.engine import train_loop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--synthetic", type=str, default="true")
    parser.add_argument("--max_steps", type=int, default=50)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.synthetic.lower() in ("true","1","yes"):
        cfg["dataset"] = "synthetic"
    os.makedirs(cfg["outputs"], exist_ok=True)
    train_loop(cfg, max_steps=args.max_steps)

if __name__ == "__main__":
    main()

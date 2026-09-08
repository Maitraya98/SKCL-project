"""
inspect_cifar_lt.py

Run this on TinyGPU (where CIFAR-10/100 are already cached under $WORK/data)
to verify the long-tailed construction against real data and generate the
class-distribution figure for the report's Reproduction section.

Usage:
    python3 scripts/inspect_cifar_lt.py --root $WORK/data --dataset cifar100 --imb_factor 0.01
    python3 scripts/inspect_cifar_lt.py --root $WORK/data --dataset cifar10  --imb_factor 0.01
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "datasets"))
from cifar_lt import IMBALANCECIFAR10, IMBALANCECIFAR100, classify_many_medium_few  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="e.g. $WORK/data")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    parser.add_argument("--imb_factor", type=float, default=0.01, help="1/beta, e.g. 0.01 for beta=100")
    parser.add_argument("--imb_type", default="exp")
    parser.add_argument("--out_dir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cls = IMBALANCECIFAR100 if args.dataset == "cifar100" else IMBALANCECIFAR10

    train_set = cls(
        root=args.root, imb_type=args.imb_type, imb_factor=args.imb_factor,
        train=True, download=False,
    )
    counts = train_set.get_cls_num_list()
    print(f"{args.dataset}-LT (beta={1/args.imb_factor:.0f}): "
          f"max={counts[0]}, min={counts[-1]}, total={sum(counts)}")

    if args.dataset == "cifar100":
        many, medium, few = classify_many_medium_few(counts)
        print(f"Many/Medium/Few split: {len(many)}/{len(medium)}/{len(few)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(counts)), counts)
        ax.set_yscale("log")
        ax.set_xlabel("Class index (sorted by frequency)")
        ax.set_ylabel("Training images (log scale)")
        ax.set_title(f"{args.dataset.upper()}-LT distribution (beta={1/args.imb_factor:.0f})")
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"{args.dataset}_lt_beta{int(1/args.imb_factor)}_distribution.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved distribution figure to {out_path}")
    except ImportError:
        print("matplotlib not available -- skipping figure, counts printed above only.")


if __name__ == "__main__":
    main()

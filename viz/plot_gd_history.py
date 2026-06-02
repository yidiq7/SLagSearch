"""Plot GD training history.

Usage:
    python -m viz.plot_gd_history <path>
    python -m viz.plot_gd_history ~/Downloads/GD_history.txt
    python -m viz.plot_gd_history gd_runs/gd_run1_step2000.pkl

<path> can be either:
  - a text log captured from gradient_descent.py stdout
  - a *.pkl checkpoint produced by gradient_descent.py (uses its 'history' field)

Output: a two-panel figure
  top:    lag_loss, spec_loss, total loss (lag + spec)
  bottom: lag_fit,  spec_fit,  total fitness (lag * spec)

The total in each panel is drawn as a thicker black line; lag/spec are
in the same accent colors (steelblue/orange) across the two panels so
trends across rows are easy to compare.
"""
import argparse
import os
import pickle
import re

import numpy as np
import matplotlib.pyplot as plt


_STEP_RE = re.compile(
    r"step\s+(\d+)\s*\|\s*loss\s+(\S+)\s*\|\s*lag_loss\s+(\S+)\s*\|\s*spec_loss\s+(\S+)"
    r"\s*\|\s*lag_fit\s+(\S+)\s*\|\s*spec_fit\s+(\S+)"
)
_INIT_RE = re.compile(
    r"initial\s*\|\s*loss\s+(\S+)\s*\|\s*lag_loss\s+(\S+)\s*\|\s*spec_loss\s+(\S+)"
    r"\s*\|\s*lag_fit\s+(\S+)\s*\|\s*spec_fit\s+(\S+)"
)


def parse_text_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = _INIT_RE.search(line)
            if m:
                rows.append((0, *[float(x) for x in m.groups()]))
                continue
            m = _STEP_RE.search(line)
            if m:
                step = int(m.group(1))
                vals = [float(x) for x in m.groups()[1:]]
                rows.append((step, *vals))
    if not rows:
        raise SystemExit(f"No history rows parsed from {path}")
    arr = np.array(rows, dtype=float)
    return {
        "step": arr[:, 0].astype(int),
        "loss": arr[:, 1],
        "lag_loss": arr[:, 2],
        "spec_loss": arr[:, 3],
        "lag_fit": arr[:, 4],
        "spec_fit": arr[:, 5],
    }


def parse_pickle(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    history = d.get("history", d) if isinstance(d, dict) else d
    return {
        "step": np.array([h["step"] for h in history]),
        "loss": np.array([h["loss"] for h in history]),
        "lag_loss": np.array([h["lag_loss"] for h in history]),
        "spec_loss": np.array([h["spec_loss"] for h in history]),
        "lag_fit": np.array([h["lag_fit"] for h in history]),
        "spec_fit": np.array([h["spec_fit"] for h in history]),
    }


def plot_history(h, loss_path, fit_path, title=None):
    total_fit = h["lag_fit"] * h["spec_fit"]
    steps = h["step"]
    title_suffix = f" -- {title}" if title else ""

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, h["lag_loss"], label="lag_loss", color="skyblue", alpha=0.7)
    ax.plot(steps, h["spec_loss"], label="spec_loss", color="orange", alpha=0.7)
    ax.plot(steps, h["loss"], label="total loss", color="black", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"GD training loss{title_suffix}")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"wrote {loss_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, h["lag_fit"], label="lag_fit", color="skyblue", alpha=0.7)
    ax.plot(steps, h["spec_fit"], label="spec_fit", color="orange", alpha=0.7)
    ax.plot(steps, total_fit, label="lag_fit * spec_fit", color="black", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Fitness")
    ax.set_title(f"GD training fitness{title_suffix}")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(fit_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fit_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="text log or .pkl checkpoint")
    parser.add_argument("--out_prefix",
                        help="output prefix; writes <prefix>_loss.png and "
                             "<prefix>_fitness.png. Default: <input> with "
                             "extension stripped, alongside the input.")
    parser.add_argument("--title", help="figure title suffix")
    parser.add_argument("--max_step", type=int, default=None,
                        help="Drop rows with step > max_step (useful when "
                             "the run plateaus early and the tail is noise).")
    args = parser.parse_args()

    if args.path.endswith(".pkl"):
        h = parse_pickle(args.path)
    else:
        h = parse_text_log(args.path)

    if args.max_step is not None:
        mask = h["step"] <= args.max_step
        h = {k: v[mask] for k, v in h.items()}
        print(f"Truncated to step <= {args.max_step}")

    print(f"Parsed {len(h['step'])} history rows")
    print(f"  final: loss={h['loss'][-1]:.6f} "
          f"lag_fit={h['lag_fit'][-1]:.4f} spec_fit={h['spec_fit'][-1]:.4f} "
          f"total_fit={h['lag_fit'][-1] * h['spec_fit'][-1]:.4f}")

    prefix = args.out_prefix or os.path.splitext(args.path)[0]
    plot_history(h, f"{prefix}_loss.png", f"{prefix}_fitness.png", title=args.title)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# /// script
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "tabulate",
# ]
# ///
"""
Generate a small, one-page Markdown report comparing two test.log files.

This script expects logs that contain lines like:
  Val result: mIoU/mAcc/allAcc 0.5265/0.5763/0.9447
  Class_0 - drivable_surface Result: iou/accuracy 0.9609/0.9815

If available, it also compares confusion matrices saved by PTv3 testing
(`confusion_matrix.npy` or `confusion_matrix.csv` placed next to `test.log`).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r"Unable to import Axes3D\..*")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class EvalResult:
    miou: float
    macc: float
    allacc: float
    classes: pd.DataFrame  # columns: class_id, class_name, iou, acc


VAL_RE = re.compile(r"Val result: mIoU/mAcc/allAcc\s+([0-9.]+)/([0-9.]+)/([0-9.]+)")
CLS_RE = re.compile(
    r"Class_(\d+)\s*-\s*(.*?)\s+Result:\s+iou/accuracy\s+([0-9.]+)/([0-9.]+)"
)


def parse_log(path: Path) -> EvalResult:
    text = path.read_text(errors="ignore").splitlines()

    best: EvalResult | None = None

    cur_miou = cur_macc = cur_allacc = None
    cur_rows: list[tuple[int, str, float, float]] = []

    for line in text:
        m = VAL_RE.search(line)
        if m:
            cur_miou, cur_macc, cur_allacc = map(float, m.groups())
            cur_rows = []
            continue

        m = CLS_RE.search(line)
        if m and cur_miou is not None:
            cid = int(m.group(1))
            cname = m.group(2).strip()
            iou = float(m.group(3))
            acc = float(m.group(4))
            cur_rows.append((cid, cname, iou, acc))

            # Heuristic: if we have a full set of classes, keep it as latest.
            # (Most of our logs are 26 classes: 0..25.)
            if len({r[0] for r in cur_rows}) >= 26:
                df = pd.DataFrame(cur_rows, columns=["class_id", "class_name", "iou", "acc"])
                df = df.sort_values("class_id").drop_duplicates("class_id", keep="last")
                if len(df) >= 26:
                    best = EvalResult(
                        miou=float(cur_miou),
                        macc=float(cur_macc),
                        allacc=float(cur_allacc),
                        classes=df.reset_index(drop=True),
                    )

    if best is None:
        raise RuntimeError(
            f"Failed to find a complete evaluation block in {path}. "
            "Expected a 'Val result:' line followed by per-class lines."
        )
    return best


def format_delta(x: float, digits: int = 4) -> str:
    s = f"{x:+.{digits}f}"
    return s


def _read_confusion_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path)
    # Expected format:
    #   gt\pred, <cls0>, <cls1>, ...
    #   <row_name>, 123, 456, ...
    if df.shape[1] < 2:
        raise RuntimeError(f"Invalid confusion CSV (need >=2 columns): {path}")
    class_names = [str(c) for c in df.columns[1:]]
    data = df.iloc[:, 1:].to_numpy(dtype=np.int64, copy=False)
    if data.shape[0] != len(class_names) or data.shape[1] != len(class_names):
        raise RuntimeError(
            f"Invalid confusion CSV shape (expected square {len(class_names)}x{len(class_names)}): "
            f"got {data.shape} in {path}"
        )
    return data, class_names


def load_confusion_from_run(run_log: Path, explicit: Path | None = None) -> tuple[np.ndarray, list[str]] | None:
    """
    Try to load the confusion matrix for a run.

    Priority:
      1) explicit path (if provided): file (.npy/.csv) or directory
      2) next to the run log: confusion_matrix.npy
      3) next to the run log: confusion_matrix.csv
    """
    if explicit is not None:
        if explicit.is_dir():
            run_dir = explicit
            npy = run_dir / "confusion_matrix.npy"
            if npy.exists():
                return np.load(npy).astype(np.int64, copy=False), []
            csv_path = run_dir / "confusion_matrix.csv"
            if csv_path.exists():
                return _read_confusion_csv(csv_path)
            raise FileNotFoundError(f"Could not find confusion matrix in directory: {explicit}")

        if explicit.suffix == ".npy":
            return np.load(explicit).astype(np.int64, copy=False), []
        if explicit.suffix == ".csv":
            return _read_confusion_csv(explicit)
        raise ValueError(f"Unsupported confusion matrix file type: {explicit}")

    run_dir = run_log.parent
    npy = run_dir / "confusion_matrix.npy"
    if npy.exists():
        return np.load(npy).astype(np.int64, copy=False), []
    csv_path = run_dir / "confusion_matrix.csv"
    if csv_path.exists():
        return _read_confusion_csv(csv_path)
    return None


def confusion_metrics(cm: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    n = int(cm.shape[0])
    if not class_names:
        class_names = [str(i) for i in range(n)]
    if cm.shape != (n, n):
        raise RuntimeError(f"Confusion matrix must be square, got {cm.shape}")

    tp = np.diag(cm).astype(np.float64)
    row_sum = cm.sum(axis=1).astype(np.float64)
    col_sum = cm.sum(axis=0).astype(np.float64)
    fn = row_sum - tp
    fp = col_sum - tp

    precision = np.zeros(n, dtype=np.float64)
    recall = np.zeros(n, dtype=np.float64)
    f1 = np.zeros(n, dtype=np.float64)

    prec_denom = tp + fp
    rec_denom = tp + fn
    p_mask = prec_denom != 0
    r_mask = rec_denom != 0
    precision[p_mask] = tp[p_mask] / prec_denom[p_mask]
    recall[r_mask] = tp[r_mask] / rec_denom[r_mask]

    pr_sum = precision + recall
    f_mask = pr_sum != 0
    f1[f_mask] = 2.0 * precision[f_mask] * recall[f_mask] / pr_sum[f_mask]

    return pd.DataFrame(
        {
            "class_id": list(range(n)),
            "class_name": class_names,
            "support_gt": row_sum.astype(np.int64),
            "support_pred": col_sum.astype(np.int64),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def confusion_gt_normalized(cm: np.ndarray) -> np.ndarray:
    row_sum = cm.sum(axis=1).astype(np.float64)
    out = np.zeros_like(cm, dtype=np.float64)
    nonzero = row_sum != 0
    out[nonzero] = cm[nonzero] / row_sum[nonzero, None]
    return out


def top_confusion_pair_deltas(
    cm_base: np.ndarray,
    cm_aug: np.ndarray,
    class_names: list[str],
    top_k: int = 12,
    direction: str = "increase",
) -> pd.DataFrame:
    gt_norm_base = confusion_gt_normalized(cm_base)
    gt_norm_aug = confusion_gt_normalized(cm_aug)
    delta = gt_norm_aug - gt_norm_base
    n = int(delta.shape[0])
    np.fill_diagonal(delta, 0.0)

    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pairs.append(
                {
                    "gt": class_names[i] if i < len(class_names) else str(i),
                    "pred": class_names[j] if j < len(class_names) else str(j),
                    "baseline": float(gt_norm_base[i, j]),
                    "augmented": float(gt_norm_aug[i, j]),
                    "delta": float(delta[i, j]),
                }
            )
    df = pd.DataFrame(pairs)
    if direction == "decrease":
        return df.sort_values("delta", ascending=True).head(top_k).reset_index(drop=True)
    return df.sort_values("delta", ascending=False).head(top_k).reset_index(drop=True)


def plot_confusion_delta_heatmap(delta: np.ndarray, class_names: list[str], outpath: Path) -> None:
    n = int(delta.shape[0])
    vmax = float(np.nanmax(np.abs(delta))) if delta.size else 0.0
    if not math.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(12.5, 10.8), constrained_layout=True)
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title("GT-normalized confusion Δ (augmented - baseline)")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")

    labels = class_names if class_names and len(class_names) == n else [str(i) for i in range(n)]
    ticks = list(range(n))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Δ probability")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_confusion_heatmap(cm_gt_norm: np.ndarray, class_names: list[str], outpath: Path, title: str) -> None:
    n = int(cm_gt_norm.shape[0])
    vmax = float(np.nanmax(cm_gt_norm)) if cm_gt_norm.size else 0.0
    if not math.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    vmax = min(1.0, vmax)

    fig, ax = plt.subplots(figsize=(12.5, 10.8), constrained_layout=True)
    im = ax.imshow(cm_gt_norm, cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")

    labels = class_names if class_names and len(class_names) == n else [str(i) for i in range(n)]
    ticks = list(range(n))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Probability")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_delta_bar(df: pd.DataFrame, outpath: Path, metric: str = "delta_iou") -> None:
    d = df.sort_values(metric, ascending=True).copy()
    colors = ["#d1495b" if v < 0 else "#2a9d8f" for v in d[metric].tolist()]

    fig_h = max(6, 0.22 * len(d) + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_h), constrained_layout=True)
    ax.barh(d["class_name"], d[metric], color=colors)
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_xlabel(metric.replace("_", " ").upper())
    ax.set_title("Per-class change (augmented - baseline)")

    # Annotate bars with values.
    for y, v in enumerate(d[metric].tolist()):
        ax.text(
            v + (0.003 if v >= 0 else -0.003),
            y,
            f"{v:+.3f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
            color="#222222",
        )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 7.0), constrained_layout=True)
    ax.scatter(df["iou_baseline"], df["iou_aug"], s=55, alpha=0.85, color="#264653")

    # y=x reference
    lo = min(df["iou_baseline"].min(), df["iou_aug"].min(), 0.0)
    hi = max(df["iou_baseline"].max(), df["iou_aug"].max(), 1.0)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="#777777")

    # Label only the most-changed classes to keep the plot readable.
    label_df = df.loc[df["delta_iou"].abs() >= 0.05].copy()
    for _, r in label_df.iterrows():
        ax.text(
            r["iou_baseline"] + 0.008,
            r["iou_aug"] + 0.008,
            r["class_name"],
            fontsize=8,
            color="#111111",
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("IoU (baseline, no small-object aug)")
    ax.set_ylabel("IoU (augmented)")
    ax.set_title("Per-class IoU: baseline vs augmented")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def resolve_test_log(path: Path) -> Path:
    """Accept either a run directory or a direct path to `test.log`."""
    if path.is_dir():
        p = path / "test.log"
        if p.exists():
            return p
        raise FileNotFoundError(f"Expected `test.log` inside directory: {path}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Baseline run directory (containing `test.log`) or `test.log` path",
    )
    ap.add_argument(
        "--aug",
        type=Path,
        required=True,
        help="Augmented run directory (containing `test.log`) or `test.log` path",
    )
    ap.add_argument(
        "--baseline-cm",
        type=Path,
        default=None,
        help="Optional confusion matrix path (file or directory) for baseline run. "
        "If omitted, tries <baseline_dir>/confusion_matrix.(npy|csv).",
    )
    ap.add_argument(
        "--aug-cm",
        type=Path,
        default=None,
        help="Optional confusion matrix path (file or directory) for augmented run. "
        "If omitted, tries <aug_dir>/confusion_matrix.(npy|csv).",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    baseline_log = resolve_test_log(args.baseline)
    aug_log = resolve_test_log(args.aug)

    baseline = parse_log(baseline_log)
    aug = parse_log(aug_log)

    df = baseline.classes.merge(
        aug.classes,
        on=["class_id", "class_name"],
        how="outer",
        suffixes=("_baseline", "_aug"),
    ).sort_values("class_id")
    for c in ["iou_baseline", "acc_baseline", "iou_aug", "acc_aug"]:
        df[c] = df[c].fillna(0.0)

    df["delta_iou"] = df["iou_aug"] - df["iou_baseline"]
    df["delta_acc"] = df["acc_aug"] - df["acc_baseline"]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot1 = out_dir / "delta_iou_bar.png"
    plot2 = out_dir / "iou_scatter.png"
    plot_delta_bar(df, plot1, metric="delta_iou")
    plot_scatter(df, plot2)

    # Confusion matrix comparison (optional, if the test run saved it).
    cm_base = load_confusion_from_run(baseline_log, explicit=args.baseline_cm)
    cm_aug = load_confusion_from_run(aug_log, explicit=args.aug_cm)
    confusion_ok = cm_base is not None and cm_aug is not None
    base_heatmap_path = out_dir / "confusion_baseline_heatmap.png"
    aug_heatmap_path = out_dir / "confusion_augmented_heatmap.png"
    delta_heatmap_path = out_dir / "confusion_delta_heatmap.png"
    if confusion_ok:
        cm_base_arr, cm_base_names = cm_base
        cm_aug_arr, cm_aug_names = cm_aug
        class_names = (
            cm_base_names
            or cm_aug_names
            or baseline.classes.sort_values("class_id")["class_name"].tolist()
        )

        base_m = confusion_metrics(cm_base_arr, class_names).add_prefix("baseline_")
        aug_m = confusion_metrics(cm_aug_arr, class_names).add_prefix("aug_")
        cm_metrics = base_m.merge(
            aug_m,
            left_on=["baseline_class_id", "baseline_class_name"],
            right_on=["aug_class_id", "aug_class_name"],
            how="outer",
        )
        cm_metrics = cm_metrics.rename(
            columns={
                "baseline_class_id": "class_id",
                "baseline_class_name": "class_name",
            }
        )
        cm_metrics["delta_precision"] = cm_metrics["aug_precision"] - cm_metrics["baseline_precision"]
        cm_metrics["delta_recall"] = cm_metrics["aug_recall"] - cm_metrics["baseline_recall"]
        cm_metrics["delta_f1"] = cm_metrics["aug_f1"] - cm_metrics["baseline_f1"]
        cm_metrics_out = cm_metrics[
            [
                "class_id",
                "class_name",
                "baseline_support_gt",
                "aug_support_gt",
                "baseline_precision",
                "aug_precision",
                "delta_precision",
                "baseline_recall",
                "aug_recall",
                "delta_recall",
                "baseline_f1",
                "aug_f1",
                "delta_f1",
            ]
        ].copy()
        cm_metrics_out.to_csv(out_dir / "confusion_metrics_delta.csv", index=False)

        cm_base_norm = confusion_gt_normalized(cm_base_arr)
        cm_aug_norm = confusion_gt_normalized(cm_aug_arr)
        plot_confusion_heatmap(cm_base_norm, class_names, base_heatmap_path, "GT-normalized confusion (baseline)")
        plot_confusion_heatmap(cm_aug_norm, class_names, aug_heatmap_path, "GT-normalized confusion (augmented)")

        delta_cm = cm_aug_norm - cm_base_norm
        plot_confusion_delta_heatmap(delta_cm, class_names, delta_heatmap_path)

        top_inc = top_confusion_pair_deltas(cm_base_arr, cm_aug_arr, class_names, top_k=12, direction="increase")
        top_dec = top_confusion_pair_deltas(cm_base_arr, cm_aug_arr, class_names, top_k=12, direction="decrease")
        top_pairs = pd.concat(
            [top_inc.assign(direction="increase"), top_dec.assign(direction="decrease")],
            ignore_index=True,
        )
        top_pairs.to_csv(out_dir / "top_confusion_pair_deltas.csv", index=False)

    # Tables
    overall = pd.DataFrame(
        [
            ["mIoU", baseline.miou, aug.miou, aug.miou - baseline.miou],
            ["mAcc", baseline.macc, aug.macc, aug.macc - baseline.macc],
            ["allAcc", baseline.allacc, aug.allacc, aug.allacc - baseline.allacc],
        ],
        columns=["metric", "baseline", "augmented", "delta"],
    )

    top_gain = (
        df.loc[df["delta_iou"] > 0]
        .sort_values("delta_iou", ascending=False)
        .head(6)[["class_name", "iou_baseline", "iou_aug", "delta_iou"]]
    )
    top_drop = (
        df.loc[df["delta_iou"] < 0]
        .sort_values("delta_iou", ascending=True)
        .head(6)[["class_name", "iou_baseline", "iou_aug", "delta_iou"]]
    )

    # Keep the report "one-page" by focusing on IoU (primary metric for segmentation),
    # while still exporting the full IoU+accuracy deltas to CSV.
    per_class_iou = df[
        ["class_id", "class_name", "iou_baseline", "iou_aug", "delta_iou"]
    ].copy()

    def md_table(frame: pd.DataFrame, float_cols: list[str]) -> str:
        f = frame.copy()
        for c in float_cols:
            if c in f.columns:
                if c.startswith("delta_"):
                    f[c] = f[c].map(lambda x: format_delta(float(x), digits=4))
                else:
                    f[c] = f[c].map(lambda x: f"{float(x):.4f}")
        return f.to_markdown(index=False)

    report = []
    report.append("# Small-object augmentation impact (per-class)\n")
    report.append("**Logs compared**\n")
    report.append(f"- Baseline (no small-object aug): `{baseline_log}`\n")
    report.append(f"- Augmented: `{aug_log}`\n")
    report.append("\n## Overall metrics\n")
    report.append(md_table(overall, ["baseline", "augmented", "delta"]))
    report.append("\n## Takeaways\n")
    report.append(
        f"- Overall: mIoU {baseline.miou:.4f} -> {aug.miou:.4f} ({format_delta(aug.miou - baseline.miou, digits=4)})"
    )
    report.append(
        f"- Biggest gain: {top_gain.iloc[0]['class_name']} IoU {top_gain.iloc[0]['iou_baseline']:.4f} -> {top_gain.iloc[0]['iou_aug']:.4f} ({format_delta(float(top_gain.iloc[0]['delta_iou']), digits=4)})"
        if len(top_gain) else "- Biggest gain: (none)"
    )
    report.append(
        f"- Biggest drop: {top_drop.iloc[0]['class_name']} IoU {top_drop.iloc[0]['iou_baseline']:.4f} -> {top_drop.iloc[0]['iou_aug']:.4f} ({format_delta(float(top_drop.iloc[0]['delta_iou']), digits=4)})"
        if len(top_drop) else "- Biggest drop: (none)"
    )
    zero_both = df.loc[(df["iou_baseline"] == 0) & (df["iou_aug"] == 0), "class_name"].tolist()
    if zero_both:
        report.append(f"- Still 0 IoU in both runs: {', '.join(zero_both)}")
    report.append("\n## Biggest per-class IoU changes\n")
    report.append("**Top gains**\n")
    report.append(md_table(top_gain, ["iou_baseline", "iou_aug", "delta_iou"]))
    report.append("\n**Top drops**\n")
    report.append(md_table(top_drop, ["iou_baseline", "iou_aug", "delta_iou"]))
    report.append("\n## Plots\n")
    report.append(f"![Per-class IoU delta]({plot1.name})\n")
    report.append(f"![Baseline vs augmented IoU scatter]({plot2.name})\n")

    if confusion_ok:
        report.append("\n## Confusion matrix (GT-normalized)\n")
        report.append(
            f"- Baseline: `{baseline_log.parent}` (expects `confusion_matrix.npy`/`.csv`)\n"
            f"- Augmented: `{aug_log.parent}` (expects `confusion_matrix.npy`/`.csv`)\n"
        )
        # Show precision deltas (often useful to understand FP spillover).
        top_prec_gain = cm_metrics_out.sort_values("delta_precision", ascending=False).head(6)[
            ["class_name", "baseline_precision", "aug_precision", "delta_precision"]
        ]
        top_prec_drop = cm_metrics_out.sort_values("delta_precision", ascending=True).head(6)[
            ["class_name", "baseline_precision", "aug_precision", "delta_precision"]
        ]
        report.append("\n**Precision deltas (from confusion matrix)**\n")
        report.append("Top gains:\n")
        report.append(md_table(top_prec_gain, ["baseline_precision", "aug_precision", "delta_precision"]))
        report.append("\nTop drops:\n")
        report.append(md_table(top_prec_drop, ["baseline_precision", "aug_precision", "delta_precision"]))

        top_pairs_md = top_pairs.copy()
        for c in ["baseline", "augmented", "delta"]:
            top_pairs_md[c] = top_pairs_md[c].map(lambda x: f"{float(x):+.6f}" if c == "delta" else f"{float(x):.6f}")
        report.append("\n**Top GT→Pred confusion changes (probability)**\n")
        report.append(top_pairs_md.to_markdown(index=False))
        report.append("\n## Confusion plot\n")
        report.append(f"![GT-normalized confusion (baseline)]({base_heatmap_path.name})\n")
        report.append(f"![GT-normalized confusion (augmented)]({aug_heatmap_path.name})\n")
        report.append(f"![GT-normalized confusion delta heatmap]({delta_heatmap_path.name})\n")
        report.append("\nFull confusion-matrix-derived deltas: `confusion_metrics_delta.csv`.\n")
        report.append("Top confusion-pair deltas: `top_confusion_pair_deltas.csv`.\n")
    else:
        report.append("\n## Confusion matrix\n")
        report.append(
            "- Confusion matrices not found for one or both runs. "
            "Expected `confusion_matrix.npy` or `confusion_matrix.csv` next to each `test.log`.\n"
        )

    report.append("\n## Per-class IoU (full list)\n")
    report.append(md_table(per_class_iou, ["iou_baseline", "iou_aug", "delta_iou"]))
    report.append(
        "\nFull per-class IoU+accuracy deltas: `per_class_delta.csv` (same folder as this report).\n"
    )
    report.append("")

    (out_dir / "report.md").write_text("\n".join(report))

    # Also write a CSV for convenience.
    df.to_csv(out_dir / "per_class_delta.csv", index=False)

    print(f"Wrote: {out_dir/'report.md'}")
    print(f"Wrote: {plot1}")
    print(f"Wrote: {plot2}")
    print(f"Wrote: {out_dir/'per_class_delta.csv'}")
    if confusion_ok:
        print(f"Wrote: {out_dir/'confusion_metrics_delta.csv'}")
        print(f"Wrote: {out_dir/'top_confusion_pair_deltas.csv'}")
        print(f"Wrote: {base_heatmap_path}")
        print(f"Wrote: {aug_heatmap_path}")
        print(f"Wrote: {delta_heatmap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

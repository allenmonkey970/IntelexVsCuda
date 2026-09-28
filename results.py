"""Summarise benchmark runs into tables and charts.

    python results.py                 # results/runs.csv  -> results/summary.{csv,md} + charts
    python results.py --legacy        # PerformanceSpeeds.csv (old single-run, training-only
                                      # numbers) -> results/legacy_summary.md

Speedup = BareBones mean time / implementation mean time, so > 1 means faster than
plain scikit-learn and < 1 means slower.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEYS = ["Dataset", "Algorithm", "Implementation"]
IMPL_ORDER = ["BareBones", "Intelex", "CuPY"]
TIMES = ["transfer_ms", "fit_ms", "predict_ms", "total_ms"]
METRICS = ["accuracy", "f1_macro"]


def fmt_ms(mean, std=None):
    if pd.isna(mean):
        return "–"
    # Keep a decimal for small values so sub-10 ms predict/transfer times are not shown as 0.
    spec = ",.1f" if mean < 100 else ",.0f"
    if std is None or pd.isna(std):
        return f"{mean:{spec}}"
    return f"{mean:{spec}} ± {std:{spec}}"


def fmt_x(x):
    return "–" if pd.isna(x) else f"{x:.2f}×"


def speedups(summary, col):
    """BareBones mean / implementation mean for each (Dataset, Algorithm)."""
    base = summary[summary["Implementation"] == "BareBones"].set_index(["Dataset", "Algorithm"])[col]
    idx = summary.set_index(["Dataset", "Algorithm"]).index
    return base.reindex(idx).to_numpy() / summary[col].to_numpy()


def summarise(runs):
    agg = {c: ["mean", "std"] for c in TIMES + METRICS}
    summary = runs.groupby(KEYS).agg(agg)
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary["runs"] = runs.groupby(KEYS).size()
    summary["Backend"] = runs.groupby(KEYS)["Backend"].first()
    summary = summary.reset_index()
    summary["speedup_fit"] = speedups(summary, "fit_ms_mean")
    summary["speedup_total"] = speedups(summary, "total_ms_mean")
    summary["Implementation"] = pd.Categorical(summary["Implementation"], IMPL_ORDER, ordered=True)
    # Keep models in the order they were run (the registry order), not alphabetical.
    model_order = list(dict.fromkeys(runs["Algorithm"]))
    summary["Algorithm"] = pd.Categorical(summary["Algorithm"], model_order, ordered=True)
    return summary.sort_values(KEYS).reset_index(drop=True)


def markdown(summary):
    n_runs = f"{summary['runs'].min()}–{summary['runs'].max()}" \
        if summary["runs"].nunique() > 1 else str(summary["runs"].iloc[0])
    lines = [
        "# Benchmark summary",
        "",
        f"Times are mean ± standard deviation over {n_runs} measured runs (after warm-up), in milliseconds.",
        "Speedup = BareBones mean / implementation mean (> 1 is faster). "
        "Accuracy and macro F1 are on the held-out test split.",
        "",
        "## Speedup factors vs BareBones (end-to-end: transfer + fit + predict)",
        "",
        "| Dataset | Model | Intelex | CuPy | Δ macro F1 Intelex | Δ macro F1 CuPy |",
        "|---|---|---|---|---|---|",
    ]
    for (dataset, algo), g in summary.groupby(["Dataset", "Algorithm"], sort=False, observed=True):
        g = g.set_index("Implementation")
        base_f1 = g["f1_macro_mean"].get("BareBones", np.nan)

        def cell(impl, col):
            return g[col].get(impl, np.nan)

        lines.append(
            f"| {dataset} | {algo} | {fmt_x(cell('Intelex', 'speedup_total'))} | {fmt_x(cell('CuPY', 'speedup_total'))} "
            f"| {cell('Intelex', 'f1_macro_mean') - base_f1:+.3f} | {cell('CuPY', 'f1_macro_mean') - base_f1:+.3f} |")

    for dataset, g in summary.groupby("Dataset", sort=False):
        lines += [
            "",
            f"## {dataset}",
            "",
            "| Model | Impl | Fit ms | Predict ms | Transfer ms | Total ms | Speedup (fit) | Speedup (total) "
            "| Accuracy | Macro F1 | Backend |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in g.iterrows():
            lines.append(
                f"| {r['Algorithm']} | {r['Implementation']} "
                f"| {fmt_ms(r['fit_ms_mean'], r['fit_ms_std'])} | {fmt_ms(r['predict_ms_mean'], r['predict_ms_std'])} "
                f"| {fmt_ms(r['transfer_ms_mean'], r['transfer_ms_std'])} | {fmt_ms(r['total_ms_mean'], r['total_ms_std'])} "
                f"| {fmt_x(r['speedup_fit'])} | {fmt_x(r['speedup_total'])} "
                f"| {r['accuracy_mean']:.4f} | {r['f1_macro_mean']:.4f} | `{r['Backend']}` |")
    return "\n".join(lines) + "\n"


def charts(summary, out_dir):
    algorithms = list(dict.fromkeys(summary["Algorithm"]))
    width = 0.8 / len(IMPL_ORDER)
    for dataset, g in summary.groupby("Dataset", sort=False):
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(algorithms))
        for i, impl in enumerate(IMPL_ORDER):
            d = g[g["Implementation"] == impl].set_index("Algorithm").reindex(algorithms)
            if d["total_ms_mean"].isna().all():
                continue
            ax.bar(x + i * width, d["total_ms_mean"], width, yerr=d["total_ms_std"], capsize=3, label=impl)
        ax.set_yscale("log")
        ax.set_ylabel("End-to-end time, ms (log scale; mean ± sd)")
        ax.set_title(f"Transfer + fit + predict time for {dataset}", fontsize=14)
        ax.set_xticks(x + width * (len(IMPL_ORDER) - 1) / 2)
        ax.set_xticklabels(algorithms, rotation=30, ha="right")
        ax.legend(title="Implementation")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{dataset}_speeds.png"))
        plt.close(fig)


def legacy(path, out_path):
    df = pd.read_csv(path).dropna(subset=["Speed(milliseconds)"])
    wide = df.pivot_table(index=["Dataset", "Algorithm"], columns="Implementation",
                          values="Speed(milliseconds)", sort=False)
    lines = [
        "# Legacy results (PerformanceSpeeds.csv)",
        "",
        "Single run per cell, training time only, no warm-up, no accuracy. In these runs the",
        "Intelex patch was applied after the estimators were imported, so the Intelex column is",
        "most likely unpatched scikit-learn. Kept for reference; superseded by results/summary.md.",
        "",
        "| Dataset | Model | BareBones ms | Intelex ms | CuPy ms | Intelex speedup | CuPy speedup |",
        "|---|---|---|---|---|---|---|",
    ]
    for (dataset, algo), r in wide.iterrows():
        lines.append(f"| {dataset} | {algo} | {fmt_ms(r['BareBones'])} | {fmt_ms(r['Intelex'])} | {fmt_ms(r['CuPY'])} "
                     f"| {fmt_x(r['BareBones'] / r['Intelex'])} | {fmt_x(r['BareBones'] / r['CuPY'])} |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=os.path.join("results", "runs.csv"))
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--legacy", action="store_true", help="summarise PerformanceSpeeds.csv instead")
    args = parser.parse_args()

    if args.legacy:
        legacy("PerformanceSpeeds.csv", os.path.join(args.out_dir, "legacy_summary.md"))
        return

    summary = summarise(pd.read_csv(args.input))
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(markdown(summary))
    charts(summary, args.out_dir)
    print(f"Wrote {args.out_dir}/summary.csv, {args.out_dir}/summary.md and per-dataset charts")


if __name__ == "__main__":
    main()

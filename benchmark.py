"""Run the benchmark and append one row per measured run to results/runs.csv.

    python benchmark.py --dataset diabetes --impl all --repeats 5 --warmup 1
    python benchmark.py --dataset all --impl all            # everything
    python results.py                                        # summary tables + charts

Each implementation runs in its own Python process: patch_sklearn() patches the whole
process, so running Intelex in the same process as BareBones would contaminate one or
the other.
"""
import argparse
import csv
import os
import subprocess
import sys

# Ensure oneDAL / TBB shared libraries are on PATH for scikit-learn-intelex.
_dll_dir = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Python", "Library", "bin")
if os.path.isdir(_dll_dir) and _dll_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_dll_dir)

IMPLEMENTATIONS = ["BareBones", "Intelex", "CuPY"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="all", help="dataset name, or 'all'")
    parser.add_argument("--impl", default="all", choices=IMPLEMENTATIONS + ["all"])
    parser.add_argument("--models", nargs="+", help="subset of model names (default: all)")
    parser.add_argument("--repeats", type=int, default=5, help="measured runs per model (default: 5)")
    parser.add_argument("--warmup", type=int, default=1, help="discarded runs before measuring (default: 1)")
    parser.add_argument("--data-dir", default="data", help="folder containing the dataset CSVs")
    parser.add_argument("--out", default=os.path.join("results", "runs.csv"))
    return parser.parse_args()


def main():
    args = parse_args()

    if args.impl == "all" or args.dataset == "all":
        # Fan out to one child process per (dataset, implementation).
        from bench.data import DATASETS
        datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
        impls = IMPLEMENTATIONS if args.impl == "all" else [args.impl]
        failed = []
        for dataset in datasets:
            for impl in impls:
                cmd = [sys.executable, __file__, "--dataset", dataset, "--impl", impl,
                       "--repeats", str(args.repeats), "--warmup", str(args.warmup),
                       "--data-dir", args.data_dir, "--out", args.out]
                if args.models:
                    cmd += ["--models", *args.models]
                if subprocess.run(cmd).returncode != 0:
                    failed.append(f"{dataset}/{impl}")
        if failed:
            sys.exit(f"Failed: {', '.join(failed)}")
        return

    if args.impl == "Intelex":
        # Must happen before anything imports a scikit-learn estimator.
        from sklearnex import patch_sklearn
        patch_sklearn()

    from bench import harness, models
    from bench.data import DATASETS

    if args.dataset not in DATASETS:
        sys.exit(f"Unknown dataset {args.dataset!r}. Choose from: {', '.join(DATASETS)}")
    model_names = args.models or models.MODELS
    unknown = set(model_names) - set(models.MODELS)
    if unknown:
        sys.exit(f"Unknown model(s): {', '.join(sorted(unknown))}. Choose from: {', '.join(models.MODELS)}")

    print(f"\n=== {args.dataset} / {args.impl}: loading data...", flush=True)
    split = DATASETS[args.dataset](args.data_dir)
    print(f"    train {split[0].shape}, test {split[1].shape}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = None
        for row in harness.run(args.dataset, args.impl, split, model_names, args.repeats, args.warmup):
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if write_header:
                    writer.writeheader()
            writer.writerow(row)
            f.flush()


if __name__ == "__main__":
    main()

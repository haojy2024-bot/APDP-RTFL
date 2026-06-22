"""Aggregate final experiment metrics across independently seeded run directories."""

import argparse
import csv
import glob
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = (
    "final_accuracy,final_f1_score,final_balanced_accuracy,final_auc_roc,"
    "avg_round_time,avg_dp_noise_scale"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate per-run final metric CSV files across random seeds."
    )
    parser.add_argument(
        "--input-root",
        default="results",
        help="Root directory containing raw run directories. Default: results.",
    )
    parser.add_argument(
        "--run-pattern",
        default=None,
        help="Glob pattern below --input-root, for example baseline_emnist_seed*.",
    )
    parser.add_argument(
        "--run-dirs",
        default=None,
        help="Optional comma-separated explicit raw run directories. Overrides --run-pattern.",
    )
    parser.add_argument(
        "--input-file",
        default="baseline_final_metrics.csv",
        help="Final metric CSV file expected inside each raw run directory.",
    )
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS,
        help="Comma-separated numeric metrics to aggregate.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory for aggregate artifacts. It must not be one of the raw run directories.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Experiment Aggregate",
        help="Prefix used in aggregate chart titles.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal places written to aggregate CSV files.",
    )
    return parser.parse_args()


def _parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _as_float(value):
    if value is None or str(value).strip() == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _seed_from_run_dir(run_dir):
    match = re.search(r"seed[_-]?(\d+)", os.path.basename(os.path.normpath(run_dir)), re.IGNORECASE)
    return match.group(1) if match else ""


def _discover_run_dirs(args):
    if args.run_dirs:
        run_dirs = [os.path.abspath(path) for path in _parse_csv_list(args.run_dirs)]
    elif args.run_pattern:
        run_dirs = sorted(
            path
            for path in glob.glob(os.path.join(args.input_root, args.run_pattern))
            if os.path.isdir(path)
        )
    else:
        raise ValueError("Provide either --run-pattern or --run-dirs.")

    if not run_dirs:
        raise FileNotFoundError("No run directories matched the supplied input.")
    return run_dirs


def _read_run_rows(run_dirs, input_file, metrics):
    rows = []
    missing = []
    for run_dir in run_dirs:
        input_path = os.path.join(run_dir, input_file)
        if not os.path.isfile(input_path):
            missing.append(input_path)
            continue
        with open(input_path, "r", newline="", encoding="utf-8") as handle:
            for source_row in csv.DictReader(handle):
                method = source_row.get("method", "").strip()
                if not method:
                    continue
                row = {
                    "run_dir": os.path.abspath(run_dir),
                    "run_name": os.path.basename(os.path.normpath(run_dir)),
                    "seed": _seed_from_run_dir(run_dir),
                    "method": method,
                    "label": source_row.get("label", method),
                    "reference": source_row.get("reference", ""),
                }
                for metric in metrics:
                    row[metric] = _as_float(source_row.get(metric))
                rows.append(row)
    if missing:
        print("Skipped run directories without the requested final metric file:")
        for path in missing:
            print(f"  {path}")
    if not rows:
        raise ValueError(f"No method rows found in {input_file}.")
    return rows


def _mean_std(values):
    valid = np.asarray([value for value in values if not np.isnan(value)], dtype=float)
    if valid.size == 0:
        return math.nan, math.nan, 0
    return float(np.mean(valid)), float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0, int(valid.size)


def _aggregate_rows(seed_rows, metrics):
    grouped = {}
    for row in seed_rows:
        grouped.setdefault(row["method"], []).append(row)

    long_rows = []
    paper_rows = []
    for method, rows in grouped.items():
        paper_row = {
            "method": method,
            "label": rows[0]["label"],
            "reference": rows[0]["reference"],
            "run_count": len(rows),
        }
        for metric in metrics:
            mean, std, count = _mean_std([row[metric] for row in rows])
            long_rows.append(
                {
                    "method": method,
                    "label": rows[0]["label"],
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "n": count,
                }
            )
            paper_row[f"{metric}_mean"] = mean
            paper_row[f"{metric}_std"] = std
            paper_row[f"{metric}_n"] = count
        paper_rows.append(paper_row)
    return long_rows, paper_rows


def _write_csv(path, rows, precision):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key, value in row.items():
                if isinstance(value, float):
                    formatted[key] = "" if math.isnan(value) else f"{value:.{precision}f}"
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def _plot_metric(long_rows, metric, output_dir, title_prefix):
    rows = [row for row in long_rows if row["metric"] == metric and row["n"] > 0]
    if not rows:
        return
    labels = [row["label"] for row in rows]
    means = [row["mean"] for row in rows]
    errors = [row["std"] for row in rows]
    x_positions = np.arange(len(rows))

    plt.figure(figsize=(max(8, len(rows) * 1.35), 5))
    plt.bar(x_positions, means, yerr=errors, capsize=4, color="#3B82B6", alpha=0.85)
    plt.xticks(x_positions, labels, rotation=25, ha="right")
    plt.ylabel(metric)
    plt.title(f"{title_prefix}: {metric} (mean +/- std)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"aggregate_{metric}.png"), dpi=300)
    plt.close()


def main():
    args = parse_args()
    metrics = _parse_csv_list(args.metrics)
    run_dirs = _discover_run_dirs(args)
    output_dir = os.path.abspath(args.output_dir)
    raw_dirs = {os.path.abspath(path) for path in run_dirs}
    if output_dir in raw_dirs:
        raise ValueError("--output-dir must be separate from every raw run directory.")

    seed_rows = _read_run_rows(run_dirs, args.input_file, metrics)
    long_rows, paper_rows = _aggregate_rows(seed_rows, metrics)
    os.makedirs(output_dir, exist_ok=True)
    _write_csv(os.path.join(output_dir, "experiment_seed_metrics.csv"), seed_rows, args.precision)
    _write_csv(os.path.join(output_dir, "experiment_metric_summary.csv"), long_rows, args.precision)
    _write_csv(os.path.join(output_dir, "experiment_paper_main_table.csv"), paper_rows, args.precision)
    for metric in metrics:
        _plot_metric(long_rows, metric, output_dir, args.title_prefix)

    print(f"Aggregated {len(run_dirs)} raw runs and {len(seed_rows)} method rows.")
    print(f"Aggregate artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()

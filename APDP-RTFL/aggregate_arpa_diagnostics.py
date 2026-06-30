"""Aggregate GRAIL-FL resource/privacy diagnostics across independent runs.

This script complements aggregate_results.py.  It focuses on mechanism
evidence such as tier-level epsilon utilization, effective participation,
upload compression, and residual-feedback behavior rather than final accuracy.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TIER_METRICS = (
    "avg_epsilon_utilization,avg_historical_success_rate,avg_deadline_feasible_rate,"
    "avg_noise_multiplier,avg_upload_ratio,avg_residual_pressure,"
    "compressed_selection_count,residual_feedback_full_upload_count"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate GRAIL-FL tier/client diagnostics across run directories."
    )
    parser.add_argument("--input-root", default="results", help="Root directory containing raw run directories.")
    parser.add_argument("--run-pattern", default=None, help="Glob pattern below --input-root, e.g. grail_main_emnist_seed*.")
    parser.add_argument("--run-dirs", default=None, help="Comma-separated explicit raw run directories. Overrides --run-pattern.")
    parser.add_argument("--method-dir", default="grail_fl", help="Method subdirectory containing GRAIL-FL diagnostics.")
    parser.add_argument("--tier-file", default="tier_privacy_summary.csv", help="Tier diagnostic CSV filename.")
    parser.add_argument("--client-file", default="resource_privacy_diagnostics.csv", help="Client diagnostic CSV filename.")
    parser.add_argument("--metrics", default=DEFAULT_TIER_METRICS, help="Comma-separated tier metrics to aggregate.")
    parser.add_argument("--scenario-filter", default=None, help="Optional comma-separated scenarios to keep, e.g. full,no_partial_updates. Use baseline for non-ablation runs.")
    parser.add_argument("--tiers", default="constrained,standard,high", help="Comma-separated resource tiers to keep.")
    parser.add_argument("--require-complete", action="store_true", help="Fail if any matched run directory lacks tier diagnostics.")
    parser.add_argument("--output-dir", required=True, help="Output directory for aggregate GRAIL-FL diagnostics.")
    parser.add_argument("--title-prefix", default="GRAIL-FL Diagnostics", help="Prefix used in aggregate chart titles.")
    parser.add_argument("--precision", type=int, default=6, help="Decimal places written to aggregate CSV files.")
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


def _seed_from_path(path):
    match = re.search(r"seed[_-]?(\d+)", path, re.IGNORECASE)
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


def _scenario_from_tier_path(run_dir, tier_path, method_dir):
    relative = os.path.relpath(tier_path, run_dir)
    parts = relative.split(os.sep)
    if len(parts) >= 3 and parts[-2] == method_dir:
        return parts[-3]
    return "baseline"


def _normalize_scenarios(value):
    if not value:
        return None
    return {"baseline" if item == "" else item for item in _parse_csv_list(value)}


def _read_tier_rows(run_dirs, args, metrics, scenario_filter, tier_filter):
    rows = []
    missing = []
    for run_dir in run_dirs:
        candidates = []
        direct_path = os.path.join(run_dir, args.method_dir, args.tier_file)
        if os.path.isfile(direct_path):
            candidates.append(direct_path)
        candidates.extend(
            path
            for path in glob.glob(os.path.join(run_dir, "**", args.method_dir, args.tier_file), recursive=True)
            if os.path.isfile(path) and path not in candidates
        )
        if not candidates:
            missing.append(os.path.join(run_dir, args.method_dir, args.tier_file))
            continue
        for path in candidates:
            scenario = _scenario_from_tier_path(run_dir, path, args.method_dir)
            if scenario_filter is not None and scenario not in scenario_filter:
                continue
            with open(path, "r", newline="", encoding="utf-8") as handle:
                for source_row in csv.DictReader(handle):
                    tier = source_row.get("tier", "")
                    if tier not in tier_filter:
                        continue
                    row = {
                        "run_dir": os.path.abspath(run_dir),
                        "run_name": os.path.basename(os.path.normpath(run_dir)),
                        "seed": _seed_from_path(os.path.basename(os.path.normpath(run_dir))),
                        "scenario": scenario,
                        "method": source_row.get("method", args.method_dir),
                        "tier": tier,
                    }
                    for metric in metrics:
                        row[metric] = _as_float(source_row.get(metric))
                    rows.append(row)
    if missing:
        print("Skipped run directories without GRAIL-FL tier diagnostics:")
        for path in missing:
            print(f"  {path}")
        if args.require_complete:
            raise FileNotFoundError("Matched run directories are missing GRAIL-FL tier diagnostics; rerun without --require-complete to skip them.")
    if not rows:
        raise ValueError("No GRAIL-FL tier diagnostic rows found.")
    return rows


def _read_client_rows(run_dirs, args, scenario_filter, tier_filter):
    rows = []
    for run_dir in run_dirs:
        candidates = []
        direct_path = os.path.join(run_dir, args.method_dir, args.client_file)
        if os.path.isfile(direct_path):
            candidates.append(direct_path)
        candidates.extend(
            path
            for path in glob.glob(os.path.join(run_dir, "**", args.method_dir, args.client_file), recursive=True)
            if os.path.isfile(path) and path not in candidates
        )
        for path in candidates:
            scenario = _scenario_from_tier_path(run_dir, path, args.method_dir)
            if scenario_filter is not None and scenario not in scenario_filter:
                continue
            with open(path, "r", newline="", encoding="utf-8") as handle:
                for source_row in csv.DictReader(handle):
                    if source_row.get("tier", "") not in tier_filter:
                        continue
                    source_row["run_dir"] = os.path.abspath(run_dir)
                    source_row["run_name"] = os.path.basename(os.path.normpath(run_dir))
                    source_row["seed"] = _seed_from_path(os.path.basename(os.path.normpath(run_dir)))
                    source_row["scenario"] = scenario
                    rows.append(source_row)
    return rows


def _mean_std(values):
    valid = np.asarray([value for value in values if not np.isnan(value)], dtype=float)
    if valid.size == 0:
        return math.nan, math.nan, 0
    std = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
    return float(np.mean(valid)), std, int(valid.size)


def _aggregate_tier_rows(seed_rows, metrics):
    grouped = {}
    for row in seed_rows:
        key = (row["scenario"], row["method"], row["tier"])
        grouped.setdefault(key, []).append(row)

    long_rows = []
    paper_rows = []
    for (scenario, method, tier), rows in sorted(grouped.items()):
        paper_row = {
            "scenario": scenario,
            "method": method,
            "tier": tier,
            "run_count": len(rows),
        }
        for metric in metrics:
            mean, std, count = _mean_std([row[metric] for row in rows])
            long_rows.append({
                "scenario": scenario,
                "method": method,
                "tier": tier,
                "metric": metric,
                "mean": mean,
                "std": std,
                "n": count,
            })
            paper_row[f"{metric}_mean"] = mean
            paper_row[f"{metric}_std"] = std
            paper_row[f"{metric}_n"] = count
        paper_rows.append(paper_row)
    return long_rows, paper_rows


def _write_csv(path, rows, precision):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
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
    scenarios = sorted({row["scenario"] for row in rows})
    tiers = ["constrained", "standard", "high"]
    for scenario in scenarios:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        if not scenario_rows:
            continue
        means = []
        errors = []
        labels = []
        for tier in tiers:
            tier_rows = [row for row in scenario_rows if row["tier"] == tier]
            if tier_rows:
                labels.append(tier)
                means.append(tier_rows[0]["mean"])
                errors.append(tier_rows[0]["std"])
        if not labels:
            continue
        x_positions = np.arange(len(labels))
        plt.figure(figsize=(8, 5))
        plt.bar(x_positions, means, yerr=errors, capsize=4, color="#3B82B6", alpha=0.85)
        plt.xticks(x_positions, labels)
        plt.ylabel(metric)
        scenario_suffix = f" ({scenario})" if scenario else ""
        plt.title(f"{title_prefix}: {metric}{scenario_suffix}")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        filename_suffix = scenario if scenario else "baseline"
        plt.savefig(os.path.join(output_dir, f"aggregate_tier_{metric}_{filename_suffix}.png"), dpi=300)
        plt.close()


def main():
    args = parse_args()
    metrics = _parse_csv_list(args.metrics)
    scenario_filter = _normalize_scenarios(args.scenario_filter)
    tier_filter = set(_parse_csv_list(args.tiers))
    if not tier_filter:
        raise ValueError("--tiers must contain at least one tier.")
    run_dirs = _discover_run_dirs(args)
    output_dir = os.path.abspath(args.output_dir)
    raw_dirs = {os.path.abspath(path) for path in run_dirs}
    if output_dir in raw_dirs:
        raise ValueError("--output-dir must be separate from every raw run directory.")

    tier_rows = _read_tier_rows(run_dirs, args, metrics, scenario_filter, tier_filter)
    client_rows = _read_client_rows(run_dirs, args, scenario_filter, tier_filter)
    long_rows, paper_rows = _aggregate_tier_rows(tier_rows, metrics)

    os.makedirs(output_dir, exist_ok=True)
    _write_csv(os.path.join(output_dir, "arpa_tier_seed_metrics.csv"), tier_rows, args.precision)
    _write_csv(os.path.join(output_dir, "arpa_tier_metric_summary.csv"), long_rows, args.precision)
    _write_csv(os.path.join(output_dir, "arpa_tier_paper_table.csv"), paper_rows, args.precision)
    if client_rows:
        _write_csv(os.path.join(output_dir, "arpa_client_seed_diagnostics.csv"), client_rows, args.precision)
    for metric in metrics:
        _plot_metric(long_rows, metric, output_dir, args.title_prefix)

    print(f"Aggregated {len(run_dirs)} raw run directories and {len(tier_rows)} tier rows.")
    if client_rows:
        print(f"Copied {len(client_rows)} client diagnostic rows.")
    print(f"GRAIL-FL aggregate artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()

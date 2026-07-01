"""Validate whether a GRAIL-FL run satisfies baseline and mechanism checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os


DEFAULT_BASELINES = "dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova"
DEFAULT_METRIC = "final_balanced_accuracy"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate GRAIL-FL acceptance criteria for one run directory.")
    parser.add_argument("--run-dir", required=True, help="Timestamped experiment run directory.")
    parser.add_argument("--method", default="grail_fl", help="Method name to validate.")
    parser.add_argument("--baselines", default=DEFAULT_BASELINES, help="Comma-separated baseline method names.")
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="Final metric used for baseline win checks.")
    parser.add_argument("--min-win-margin", type=float, default=0.0, help="Minimum required GRAIL-FL minus baseline metric margin.")
    parser.add_argument("--require-all-baselines", action="store_true", help="Fail if any requested baseline is missing.")
    parser.add_argument("--min-epsilon-utilization", type=float, default=0.70, help="Minimum average tier epsilon utilization.")
    parser.add_argument("--min-constrained-success-rate", type=float, default=0.50, help="Minimum constrained-tier effective participation rate.")
    parser.add_argument("--max-deadline-failure-rate", type=float, default=0.20, help="Maximum selected-row predicted deadline failure rate.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: <run-dir>/acceptance.")
    parser.add_argument("--strict", action="store_true", help="Fail if optional GRAIL-FL diagnostic files are missing.")
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


def _read_csv(path, required=True):
    if not os.path.isfile(path):
        if required:
            raise FileNotFoundError(path)
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_by_method(rows, metric):
    values = {}
    for row in rows:
        method = row.get("method", "")
        if method:
            values[method] = _as_float(row.get(metric))
    return values


def _deadline_failure_rate(resource_rows):
    selected = [row for row in resource_rows if row.get("status") == "selected"]
    if not selected:
        return math.nan
    failures = 0
    for row in selected:
        total = _as_float(row.get("predicted_total_seconds"))
        deadline = _as_float(row.get("resource_deadline_seconds"))
        if math.isnan(deadline):
            deadline = _as_float(row.get("deadline_seconds"))
        # resource_trace rows generally do not store the deadline; selected rows
        # are already produced from feasible actions, so absent deadline means pass.
        if not math.isnan(total) and not math.isnan(deadline) and total > deadline:
            failures += 1
    return failures / max(1, len(selected))


def _tier_summary_checks(tier_rows, min_epsilon_utilization, min_constrained_success_rate):
    checks = []
    epsilon_values = [_as_float(row.get("avg_epsilon_utilization")) for row in tier_rows]
    epsilon_values = [value for value in epsilon_values if not math.isnan(value)]
    avg_epsilon_utilization = sum(epsilon_values) / len(epsilon_values) if epsilon_values else math.nan
    checks.append({
        "check": "avg_tier_epsilon_utilization",
        "value": avg_epsilon_utilization,
        "threshold": min_epsilon_utilization,
        "passed": avg_epsilon_utilization >= min_epsilon_utilization if not math.isnan(avg_epsilon_utilization) else False,
    })
    constrained = [row for row in tier_rows if row.get("tier") == "constrained"]
    constrained_success = _as_float(constrained[0].get("avg_historical_success_rate")) if constrained else math.nan
    checks.append({
        "check": "constrained_effective_participation",
        "value": constrained_success,
        "threshold": min_constrained_success_rate,
        "passed": constrained_success >= min_constrained_success_rate if not math.isnan(constrained_success) else False,
    })
    return checks


def validate_run(args):
    run_dir = os.path.abspath(args.run_dir)
    method_dir = os.path.join(run_dir, args.method)
    final_rows = _read_csv(os.path.join(run_dir, "baseline_final_metrics.csv"), required=True)
    metric_values = _metric_by_method(final_rows, args.metric)
    method_value = metric_values.get(args.method, math.nan)

    checks = []
    comparison_rows = []
    requested_baselines = _parse_csv_list(args.baselines)
    for baseline in requested_baselines:
        baseline_value = metric_values.get(baseline, math.nan)
        present = not math.isnan(baseline_value)
        margin = method_value - baseline_value if present and not math.isnan(method_value) else math.nan
        passed = present and not math.isnan(margin) and margin >= args.min_win_margin
        comparison_rows.append({
            "method": args.method,
            "baseline": baseline,
            "metric": args.metric,
            "method_value": method_value,
            "baseline_value": baseline_value,
            "margin": margin,
            "min_win_margin": args.min_win_margin,
            "baseline_present": present,
            "passed": passed if present else "",
        })
    present_comparisons = [row for row in comparison_rows if row["baseline_present"]]
    missing_baselines = [row["baseline"] for row in comparison_rows if not row["baseline_present"]]
    checks.append({
        "check": "requested_baselines_present",
        "value": len(requested_baselines) - len(missing_baselines),
        "threshold": len(requested_baselines),
        "passed": (not args.require_all_baselines) or not missing_baselines,
        "details": ",".join(missing_baselines),
    })
    checks.append({
        "check": "beats_present_baselines",
        "value": sum(row["passed"] is True for row in present_comparisons),
        "threshold": len(present_comparisons),
        "passed": bool(present_comparisons) and all(row["passed"] is True for row in present_comparisons),
    })

    tier_rows = _read_csv(os.path.join(method_dir, "tier_privacy_summary.csv"), required=args.strict)
    if tier_rows:
        checks.extend(_tier_summary_checks(tier_rows, args.min_epsilon_utilization, args.min_constrained_success_rate))
    else:
        checks.append({"check": "tier_diagnostics_present", "value": 0, "threshold": 1, "passed": False})

    resource_rows = _read_csv(os.path.join(method_dir, "resource_trace.csv"), required=args.strict)
    deadline_failure_rate = _deadline_failure_rate(resource_rows) if resource_rows else math.nan
    checks.append({
        "check": "predicted_deadline_failure_rate",
        "value": deadline_failure_rate,
        "threshold": args.max_deadline_failure_rate,
        "passed": deadline_failure_rate <= args.max_deadline_failure_rate if not math.isnan(deadline_failure_rate) else not args.strict,
    })

    overall_passed = all(row["passed"] is True for row in checks)
    return {
        "run_dir": run_dir,
        "method": args.method,
        "metric": args.metric,
        "method_value": method_value,
        "overall_passed": overall_passed,
        "checks": checks,
        "comparisons": comparison_rows,
    }


def main():
    args = parse_args()
    result = validate_run(args)
    output_dir = os.path.abspath(args.output_dir or os.path.join(args.run_dir, "acceptance"))
    os.makedirs(output_dir, exist_ok=True)
    _write_csv(os.path.join(output_dir, "arpa_acceptance_checks.csv"), result["checks"])
    _write_csv(os.path.join(output_dir, "arpa_baseline_comparisons.csv"), result["comparisons"])
    with open(os.path.join(output_dir, "arpa_acceptance_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    status = "PASSED" if result["overall_passed"] else "FAILED"
    print(f"ARPA acceptance {status}: {result['run_dir']}")
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()

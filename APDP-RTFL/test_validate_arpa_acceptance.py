import csv
import os
import unittest
import uuid
from types import SimpleNamespace

from validate_arpa_acceptance import validate_run


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ValidateArpaAcceptanceTests(unittest.TestCase):
    def _workspace_temp_dir(self):
        root = os.path.join(os.getcwd(), "results", f"_validate_arpa_test_{uuid.uuid4().hex}")
        os.makedirs(root, exist_ok=False)
        return root

    def _args(self, run_dir, **overrides):
        values = {
            "run_dir": run_dir,
            "method": "apdp_rtfl",
            "baselines": "dp_fl,dp_rtfl",
            "metric": "final_balanced_accuracy",
            "min_win_margin": 0.0,
            "require_all_baselines": False,
            "min_epsilon_utilization": 0.70,
            "min_constrained_success_rate": 0.50,
            "max_deadline_failure_rate": 0.20,
            "strict": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _make_run(self, root, apdp_score=0.8, dp_fl_score=0.5, dp_rtfl_score=0.6,
                  epsilon_utilization=0.9, constrained_success=0.75):
        _write_csv(os.path.join(root, "baseline_final_metrics.csv"), [
            {"method": "dp_fl", "final_balanced_accuracy": dp_fl_score},
            {"method": "dp_rtfl", "final_balanced_accuracy": dp_rtfl_score},
            {"method": "apdp_rtfl", "final_balanced_accuracy": apdp_score},
        ])
        _write_csv(os.path.join(root, "apdp_rtfl", "tier_privacy_summary.csv"), [
            {
                "tier": "constrained",
                "avg_epsilon_utilization": epsilon_utilization,
                "avg_historical_success_rate": constrained_success,
            },
            {
                "tier": "standard",
                "avg_epsilon_utilization": epsilon_utilization,
                "avg_historical_success_rate": 1.0,
            },
        ])
        _write_csv(os.path.join(root, "apdp_rtfl", "resource_trace.csv"), [
            {"status": "selected", "predicted_total_seconds": "1.0"},
            {"status": "not_selected", "predicted_total_seconds": "1.2"},
        ])

    def test_acceptance_passes_when_apdp_beats_baselines_and_diagnostics_pass(self):
        run_dir = self._workspace_temp_dir()
        self._make_run(run_dir)
        result = validate_run(self._args(run_dir))
        self.assertTrue(result["overall_passed"])
        checks = {row["check"]: row["passed"] for row in result["checks"]}
        self.assertTrue(checks["beats_present_baselines"])
        self.assertTrue(checks["avg_tier_epsilon_utilization"])
        self.assertTrue(checks["constrained_effective_participation"])

    def test_acceptance_fails_when_apdp_loses_to_present_baseline(self):
        run_dir = self._workspace_temp_dir()
        self._make_run(run_dir, apdp_score=0.4)
        result = validate_run(self._args(run_dir))
        self.assertFalse(result["overall_passed"])
        checks = {row["check"]: row["passed"] for row in result["checks"]}
        self.assertFalse(checks["beats_present_baselines"])

    def test_acceptance_fails_when_win_margin_is_too_small(self):
        run_dir = self._workspace_temp_dir()
        self._make_run(run_dir, apdp_score=0.61, dp_rtfl_score=0.60)
        result = validate_run(self._args(run_dir, min_win_margin=0.05))
        self.assertFalse(result["overall_passed"])
        comparison = {row["baseline"]: row for row in result["comparisons"]}
        self.assertFalse(comparison["dp_rtfl"]["passed"])

    def test_require_all_baselines_fails_when_requested_baseline_is_missing(self):
        run_dir = self._workspace_temp_dir()
        self._make_run(run_dir)
        result = validate_run(self._args(run_dir, baselines="dp_fl,missing_dp", require_all_baselines=True))
        self.assertFalse(result["overall_passed"])
        checks = {row["check"]: row for row in result["checks"]}
        self.assertFalse(checks["requested_baselines_present"]["passed"])
        self.assertEqual(checks["requested_baselines_present"]["details"], "missing_dp")

    def test_acceptance_fails_when_epsilon_utilization_is_low(self):
        run_dir = self._workspace_temp_dir()
        self._make_run(run_dir, epsilon_utilization=0.4)
        result = validate_run(self._args(run_dir))
        self.assertFalse(result["overall_passed"])
        checks = {row["check"]: row["passed"] for row in result["checks"]}
        self.assertFalse(checks["avg_tier_epsilon_utilization"])

    def test_strict_mode_requires_tier_diagnostics(self):
        run_dir = self._workspace_temp_dir()
        _write_csv(os.path.join(run_dir, "baseline_final_metrics.csv"), [
            {"method": "apdp_rtfl", "final_balanced_accuracy": 0.8},
        ])
        with self.assertRaises(FileNotFoundError):
            validate_run(self._args(run_dir, strict=True))


if __name__ == "__main__":
    unittest.main()

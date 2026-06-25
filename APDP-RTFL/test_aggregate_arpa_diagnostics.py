import csv
import os
import unittest
import uuid
from types import SimpleNamespace

from aggregate_arpa_diagnostics import (
    _aggregate_tier_rows,
    _discover_run_dirs,
    _normalize_scenarios,
    _read_client_rows,
    _read_tier_rows,
)


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


class AggregateArpaDiagnosticsTests(unittest.TestCase):
    def _workspace_temp_dir(self):
        root = os.path.join(os.getcwd(), "results", f"_aggregate_arpa_test_{uuid.uuid4().hex}")
        os.makedirs(root, exist_ok=False)
        return root

    def _args(self, **overrides):
        values = {
            "input_root": "",
            "run_pattern": None,
            "run_dirs": None,
            "method_dir": "apdp_rtfl",
            "tier_file": "tier_privacy_summary.csv",
            "client_file": "resource_privacy_diagnostics.csv",
            "require_complete": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _make_baseline_run(self, root, name="arpa_seed42"):
        run_dir = os.path.join(root, name)
        _write_csv(os.path.join(run_dir, "apdp_rtfl", "tier_privacy_summary.csv"), [
            {
                "method": "apdp_rtfl",
                "tier": "constrained",
                "avg_epsilon_utilization": "0.8",
                "avg_upload_ratio": "0.5",
            },
            {
                "method": "apdp_rtfl",
                "tier": "standard",
                "avg_epsilon_utilization": "0.9",
                "avg_upload_ratio": "1.0",
            },
        ])
        _write_csv(os.path.join(run_dir, "apdp_rtfl", "resource_privacy_diagnostics.csv"), [
            {"client_id": "client_0", "tier": "constrained", "epsilon_utilization": "0.8"},
            {"client_id": "client_1", "tier": "standard", "epsilon_utilization": "0.9"},
        ])
        return run_dir

    def _make_ablation_run(self, root, name="ablation_seed42"):
        run_dir = os.path.join(root, name)
        for scenario, value in (("full", "0.8"), ("no_partial_updates", "0.6")):
            _write_csv(os.path.join(run_dir, scenario, "apdp_rtfl", "tier_privacy_summary.csv"), [
                {
                    "method": "apdp_rtfl",
                    "tier": "constrained",
                    "avg_epsilon_utilization": value,
                    "avg_upload_ratio": "0.5",
                }
            ])
            _write_csv(os.path.join(run_dir, scenario, "apdp_rtfl", "resource_privacy_diagnostics.csv"), [
                {"client_id": "client_0", "tier": "constrained", "epsilon_utilization": value}
            ])
        return run_dir

    def test_reads_baseline_tier_and_client_rows(self):
        temp_dir = self._workspace_temp_dir()
        run_dir = self._make_baseline_run(temp_dir)
        args = self._args()
        metrics = ["avg_epsilon_utilization", "avg_upload_ratio"]
        tier_rows = _read_tier_rows([run_dir], args, metrics, None, {"constrained", "standard"})
        client_rows = _read_client_rows([run_dir], args, None, {"constrained"})
        self.assertEqual(len(tier_rows), 2)
        self.assertEqual(tier_rows[0]["scenario"], "baseline")
        self.assertEqual(len(client_rows), 1)
        self.assertEqual(client_rows[0]["tier"], "constrained")

    def test_scenario_and_tier_filters_apply_to_ablation_runs(self):
        temp_dir = self._workspace_temp_dir()
        run_dir = self._make_ablation_run(temp_dir)
        args = self._args()
        metrics = ["avg_epsilon_utilization"]
        rows = _read_tier_rows(
            [run_dir],
            args,
            metrics,
            _normalize_scenarios("full"),
            {"constrained"},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["scenario"], "full")
        self.assertAlmostEqual(rows[0]["avg_epsilon_utilization"], 0.8)

    def test_require_complete_fails_on_missing_tier_file(self):
        temp_dir = self._workspace_temp_dir()
        missing_run = os.path.join(temp_dir, "arpa_seed43")
        os.makedirs(missing_run, exist_ok=True)
        args = self._args(require_complete=True)
        with self.assertRaises(FileNotFoundError):
            _read_tier_rows([missing_run], args, ["avg_epsilon_utilization"], None, {"constrained"})

    def test_aggregation_groups_by_scenario_method_and_tier(self):
        rows = [
            {"scenario": "baseline", "method": "apdp_rtfl", "tier": "constrained", "avg_epsilon_utilization": 0.8},
            {"scenario": "baseline", "method": "apdp_rtfl", "tier": "constrained", "avg_epsilon_utilization": 1.0},
        ]
        long_rows, paper_rows = _aggregate_tier_rows(rows, ["avg_epsilon_utilization"])
        self.assertEqual(len(long_rows), 1)
        self.assertEqual(len(paper_rows), 1)
        self.assertAlmostEqual(paper_rows[0]["avg_epsilon_utilization_mean"], 0.9)
        self.assertEqual(paper_rows[0]["avg_epsilon_utilization_n"], 2)

    def test_discovers_run_directories_from_pattern(self):
        temp_dir = self._workspace_temp_dir()
        self._make_baseline_run(temp_dir, "arpa_seed42")
        os.makedirs(os.path.join(temp_dir, "not_a_run.txt"), exist_ok=True)
        args = self._args(input_root=temp_dir, run_pattern="arpa_seed*")
        run_dirs = _discover_run_dirs(args)
        self.assertEqual(len(run_dirs), 1)
        self.assertTrue(run_dirs[0].endswith("arpa_seed42"))


if __name__ == "__main__":
    unittest.main()

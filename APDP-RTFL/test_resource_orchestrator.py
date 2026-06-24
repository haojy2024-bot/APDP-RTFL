import unittest

import numpy as np

from privacy_accounting import RDPAccountant
from resource_orchestrator import (
    ResourcePrivacyOrchestrator,
    build_resource_profiles,
    mask_delta,
    parameter_bytes,
    rotating_block_mask,
)


class ResourceOrchestratorTests(unittest.TestCase):
    def test_profiles_and_snapshots_are_reproducible(self):
        first = build_resource_profiles(10, 42)
        second = build_resource_profiles(10, 42)
        self.assertEqual(first, second)
        orchestrator = ResourcePrivacyOrchestrator(first, 42, 5.0, 0.01, (1.0, 0.5), 8)
        self.assertEqual(orchestrator.snapshot(0, 1), orchestrator.snapshot(0, 1))

    def test_predicted_latency_is_monotonic(self):
        profiles = build_resource_profiles(3, 1)
        orchestrator = ResourcePrivacyOrchestrator(profiles, 1, 100.0, 0.01, (1.0, 0.5), 4)
        snapshot = orchestrator.snapshot(0, 1)
        _, _, short = orchestrator.predict_seconds(snapshot, 100, 10, 1, 0.5, 4096)
        _, _, long = orchestrator.predict_seconds(snapshot, 100, 10, 3, 1.0, 4096)
        self.assertGreater(long, short)

    def test_masks_rotate_and_preserve_only_selected_coordinates(self):
        delta = {"coef_": np.ones((2, 8)), "intercept_": np.ones(2)}
        first = rotating_block_mask(delta, 4, 0.25, 0, 1)
        second = rotating_block_mask(delta, 4, 0.25, 0, 2)
        self.assertFalse(np.array_equal(first["coef_"], second["coef_"]))
        masked = mask_delta(delta, first)
        self.assertTrue(np.all(masked["coef_"][~first["coef_"]] == 0))
        self.assertLess(parameter_bytes(masked), 1000)

    def test_plan_respects_deadline_and_privacy_ledger(self):
        profiles = build_resource_profiles(6, 4)
        orchestrator = ResourcePrivacyOrchestrator(profiles, 4, 2.0, 0.001, (1.0, 0.5, 0.25), 4)
        privacy_states = {
            idx: {"accountant": RDPAccountant(5.0, 1e-5), "base_noise_multiplier": 2.0, "sample_rate": 0.1}
            for idx in profiles
        }
        actions, _ = orchestrator.plan(
            round_num=1, sample_counts={idx: 100 for idx in profiles}, batch_size=10, base_epochs=3,
            model_bytes=1024, participation_counts=[0] * 6, contribution_scores={idx: 0.0 for idx in profiles},
            quality_scores={idx: 0.5 for idx in profiles}, privacy_states=privacy_states,
            remaining_rounds=10, target_count=3, eligible_indices=set(profiles),
        )
        self.assertLessEqual(len(actions), 3)
        for action in actions.values():
            self.assertLessEqual(action.predicted_total_seconds, 2.0)
            self.assertIsNotNone(action.noise_multiplier)

    def test_first_round_uses_cold_start_spend_and_records_reason(self):
        profiles = build_resource_profiles(5, 42)
        orchestrator = ResourcePrivacyOrchestrator(profiles, 42, 5.0, 0.01, (1.0,), 4)
        privacy_states = {
            idx: {"accountant": RDPAccountant(5.0, 1e-5), "base_noise_multiplier": 4.0, "sample_rate": 0.1}
            for idx in profiles
        }
        actions, trace = orchestrator.plan(
            round_num=1, sample_counts={idx: 100 for idx in profiles}, batch_size=10, base_epochs=1,
            model_bytes=1024, participation_counts=[0] * 5, contribution_scores={idx: 0.0 for idx in profiles},
            quality_scores={idx: 0.5 for idx in profiles}, privacy_states=privacy_states,
            remaining_rounds=50, target_count=5, eligible_indices=set(profiles),
        )
        self.assertTrue(actions)
        self.assertFalse(any(row["status"] == "deadline_or_privacy" for row in trace))
        self.assertTrue(all(action.privacy_target_increment >= 0.1 for action in actions.values()))

    def test_trace_separates_deadline_and_privacy_rejections(self):
        profiles = build_resource_profiles(3, 7)
        privacy_states = {
            idx: {"accountant": RDPAccountant(0.001, 1e-5), "base_noise_multiplier": 2.0, "sample_rate": 0.1}
            for idx in profiles
        }
        privacy_limited = ResourcePrivacyOrchestrator(profiles, 7, 5.0, 0.001, (1.0,), 4)
        _, trace = privacy_limited.plan(
            round_num=1, sample_counts={idx: 10 for idx in profiles}, batch_size=10, base_epochs=1,
            model_bytes=100, participation_counts=[0] * 3, contribution_scores={idx: 0.0 for idx in profiles},
            quality_scores={idx: 0.5 for idx in profiles}, privacy_states=privacy_states,
            remaining_rounds=1, target_count=3, eligible_indices=set(profiles),
        )
        self.assertIn("privacy_budget_infeasible", {row["status"] for row in trace})
        deadline_limited = ResourcePrivacyOrchestrator(profiles, 7, 1e-9, 0.001, (1.0,), 4)
        _, trace = deadline_limited.plan(
            round_num=1, sample_counts={idx: 10 for idx in profiles}, batch_size=10, base_epochs=1,
            model_bytes=100, participation_counts=[0] * 3, contribution_scores={idx: 0.0 for idx in profiles},
            quality_scores={idx: 0.5 for idx in profiles}, privacy_states={},
            remaining_rounds=1, target_count=3, eligible_indices=set(profiles),
        )
        self.assertIn("deadline_infeasible", {row["status"] for row in trace})


if __name__ == "__main__":
    unittest.main()

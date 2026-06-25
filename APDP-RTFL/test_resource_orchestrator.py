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

    def test_opportunity_aware_privacy_spends_more_for_scarce_windows(self):
        profiles = build_resource_profiles(2, 11)
        orchestrator = ResourcePrivacyOrchestrator(profiles, 11, 5.0, 0.001, (1.0,), 4)
        high_opportunity_state = {"accountant": RDPAccountant(5.0, 1e-5), "base_noise_multiplier": 2.0, "sample_rate": 0.1}
        low_opportunity_state = {"accountant": RDPAccountant(5.0, 1e-5), "base_noise_multiplier": 2.0, "sample_rate": 0.1}
        high_opportunity_state["accountant"].spend(0, 1, 0.1, 4.0)
        low_opportunity_state["accountant"].spend(0, 1, 0.1, 4.0)
        high_choice = orchestrator._privacy_choice(
            high_opportunity_state, 5, 20, 0.8, 0.5, "none",
            expected_opportunities=15.0, budget_utilization=0.2, privacy_boost=1.0,
            opportunity_compensation=0.0, effective_work_ratio=1.0,
        )
        low_choice = orchestrator._privacy_choice(
            low_opportunity_state, 5, 20, 0.8, 0.5, "none",
            expected_opportunities=3.0, budget_utilization=0.2, privacy_boost=1.0,
            opportunity_compensation=0.8, effective_work_ratio=1.0,
        )
        self.assertGreater(low_choice[2], high_choice[2])

    def test_trace_includes_explainable_privacy_fields(self):
        profiles = build_resource_profiles(5, 13)
        orchestrator = ResourcePrivacyOrchestrator(profiles, 13, 5.0, 0.001, (1.0, 0.5), 4)
        privacy_states = {
            idx: {"accountant": RDPAccountant(5.0, 1e-5), "base_noise_multiplier": 3.0, "sample_rate": 0.1}
            for idx in profiles
        }
        actions, trace = orchestrator.plan(
            round_num=1, sample_counts={idx: 50 for idx in profiles}, batch_size=10, base_epochs=1,
            model_bytes=512, participation_counts=[0] * 5, contribution_scores={idx: 0.1 for idx in profiles},
            quality_scores={idx: 0.7 for idx in profiles}, privacy_states=privacy_states,
            remaining_rounds=10, target_count=3, eligible_indices=set(profiles),
        )
        self.assertTrue(actions)
        selected_rows = [row for row in trace if row["status"] == "selected"]
        self.assertTrue(selected_rows)
        for field in (
            "privacy_budget_target",
            "expected_future_opportunities",
            "budget_utilization",
            "privacy_boost",
            "opportunity_compensation",
            "privacy_cap_reason",
            "selected_noise_reason",
        ):
            self.assertIn(field, selected_rows[0])

    def test_low_resource_compensation_can_be_disabled(self):
        profiles = build_resource_profiles(6, 21)
        enabled = ResourcePrivacyOrchestrator(
            profiles, 21, 5.0, 0.001, (1.0, 0.5), 4,
            enable_low_resource_compensation=True,
        )
        disabled = ResourcePrivacyOrchestrator(
            profiles, 21, 5.0, 0.001, (1.0, 0.5), 4,
            enable_low_resource_compensation=False,
        )
        common_kwargs = dict(
            round_num=1,
            sample_counts={idx: 50 for idx in profiles},
            batch_size=10,
            base_epochs=1,
            model_bytes=512,
            participation_counts=[0] * 6,
            contribution_scores={idx: 0.1 for idx in profiles},
            quality_scores={idx: 0.7 for idx in profiles},
            privacy_states={
                idx: {"accountant": RDPAccountant(5.0, 1e-5), "base_noise_multiplier": 3.0, "sample_rate": 0.1}
                for idx in profiles
            },
            remaining_rounds=10,
            target_count=4,
            eligible_indices=set(profiles),
        )
        enabled_actions, _ = enabled.plan(**common_kwargs)
        disabled_actions, _ = disabled.plan(**{
            **common_kwargs,
            "privacy_states": {
                idx: {"accountant": RDPAccountant(5.0, 1e-5), "base_noise_multiplier": 3.0, "sample_rate": 0.1}
                for idx in profiles
            },
        })
        self.assertTrue(any(action.opportunity_compensation > 0 for action in enabled_actions.values()))
        self.assertTrue(all(action.opportunity_compensation == 0 for action in disabled_actions.values()))


if __name__ == "__main__":
    unittest.main()

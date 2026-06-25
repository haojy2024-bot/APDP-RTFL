"""Auditable resource/privacy orchestration for regulated FL simulations.

The module is deliberately simulation-oriented: it predicts compute and network
time from reproducible client profiles without sleeping or relying on host
hardware.  Privacy feasibility is delegated to the caller's RDP accountant.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


RESOURCE_TIERS = ("constrained", "standard", "high")


@dataclass(frozen=True)
class ResourceProfile:
    client_idx: int
    tier: str
    compute_speed: float
    uplink_mbps: float
    rtt_ms: float
    availability: float


@dataclass(frozen=True)
class ResourceSnapshot:
    profile: ResourceProfile
    round_num: int
    compute_speed: float
    uplink_mbps: float
    rtt_ms: float
    online: bool


@dataclass(frozen=True)
class PlannedAction:
    client_idx: int
    epochs: int
    upload_ratio: float
    noise_multiplier: float | None
    predicted_compute_seconds: float
    predicted_communication_seconds: float
    predicted_total_seconds: float
    privacy_target_increment: float
    privacy_budget_target: float
    expected_future_opportunities: float
    budget_utilization: float
    privacy_boost: float
    opportunity_compensation: float
    privacy_cap_reason: str
    selected_noise_reason: str
    utility_score: float


def build_resource_profiles(num_clients: int, seed: int) -> dict[int, ResourceProfile]:
    """Build a deterministic generic regulated-industry device population."""
    rng = np.random.default_rng(seed + 811)
    tier_specs = {
        "constrained": (0.45, 10.0, 85.0, 0.88),
        "standard": (1.00, 50.0, 35.0, 0.95),
        "high": (1.60, 200.0, 12.0, 0.99),
    }
    tiers = ["constrained"] * int(np.ceil(num_clients * 0.2))
    tiers += ["standard"] * int(np.ceil(num_clients * 0.6))
    tiers += ["high"] * max(0, num_clients - len(tiers))
    tiers = tiers[:num_clients]
    rng.shuffle(tiers)
    profiles = {}
    for idx, tier in enumerate(tiers):
        speed, uplink, rtt, availability = tier_specs[tier]
        profiles[idx] = ResourceProfile(
            client_idx=idx,
            tier=tier,
            compute_speed=float(speed * rng.uniform(0.92, 1.08)),
            uplink_mbps=float(uplink * rng.uniform(0.85, 1.15)),
            rtt_ms=float(rtt * rng.uniform(0.85, 1.15)),
            availability=float(np.clip(availability, 0.01, 1.0)),
        )
    return profiles


def parameter_bytes(parameters: dict[str, np.ndarray]) -> int:
    return int(sum(np.asarray(value).nbytes for value in parameters.values()))


def rotating_block_mask(parameters: dict[str, np.ndarray], block_count: int, ratio: float,
                        client_idx: int, round_num: int) -> dict[str, np.ndarray]:
    """Select contiguous parameter blocks deterministically for reproducibility."""
    if block_count < 1:
        raise ValueError("block_count must be positive")
    if not 0 < ratio <= 1:
        raise ValueError("upload ratio must be in (0, 1]")
    total = sum(np.asarray(value).size for value in parameters.values())
    active_blocks = min(block_count, max(1, int(np.ceil(block_count * ratio))))
    start = (client_idx + round_num - 1) % block_count
    selected = {(start + offset) % block_count for offset in range(active_blocks)}
    masks, offset = {}, 0
    for key, value in parameters.items():
        array = np.asarray(value)
        flat_ids = np.arange(offset, offset + array.size)
        block_ids = np.minimum(block_count - 1, flat_ids * block_count // max(1, total))
        masks[key] = np.isin(block_ids, list(selected)).reshape(array.shape)
        offset += array.size
    return masks


def mask_delta(delta: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.where(masks[key], value, 0.0) for key, value in delta.items()}


class ResourcePrivacyOrchestrator:
    """Select deadline-feasible actions while retaining slow, compliant clients."""

    def __init__(self, profiles: dict[int, ResourceProfile], seed: int, deadline_seconds: float,
                 reference_batch_seconds: float, upload_ratios: tuple[float, ...], block_count: int,
                 enforce_tier_coverage: bool = True, minimum_initial_privacy_increment: float = 0.25,
                 enable_opportunity_privacy: bool = True, enable_budget_utilization_boost: bool = True,
                 enable_low_resource_compensation: bool = True, privacy_boost_gain: float = 0.8,
                 max_privacy_boost: float = 1.8, opportunity_compensation_weight: float = 0.65):
        self.profiles = profiles
        self.seed = seed
        self.deadline_seconds = float(deadline_seconds)
        self.reference_batch_seconds = float(reference_batch_seconds)
        self.upload_ratios = tuple(sorted({float(item) for item in upload_ratios}, reverse=True))
        self.block_count = int(block_count)
        self.enforce_tier_coverage = bool(enforce_tier_coverage)
        self.minimum_initial_privacy_increment = float(minimum_initial_privacy_increment)
        self.enable_opportunity_privacy = bool(enable_opportunity_privacy)
        self.enable_budget_utilization_boost = bool(enable_budget_utilization_boost)
        self.enable_low_resource_compensation = bool(enable_low_resource_compensation)
        self.privacy_boost_gain = float(privacy_boost_gain)
        self.max_privacy_boost = float(max_privacy_boost)
        self.opportunity_compensation_weight = float(opportunity_compensation_weight)
        if self.minimum_initial_privacy_increment <= 0:
            raise ValueError("minimum_initial_privacy_increment must be positive")
        if self.privacy_boost_gain < 0 or self.max_privacy_boost < 1.0:
            raise ValueError("privacy boost gain must be non-negative and max boost must be at least 1")
        if self.opportunity_compensation_weight < 0:
            raise ValueError("opportunity compensation weight must be non-negative")

    def snapshot(self, client_idx: int, round_num: int) -> ResourceSnapshot:
        profile = self.profiles[client_idx]
        rng = np.random.default_rng(self.seed + 100_003 * round_num + client_idx)
        speed = profile.compute_speed * rng.uniform(0.90, 1.10)
        uplink = profile.uplink_mbps * rng.uniform(0.80, 1.20)
        rtt = profile.rtt_ms * rng.uniform(0.85, 1.15)
        return ResourceSnapshot(profile, round_num, float(speed), float(uplink), float(rtt), bool(rng.random() < profile.availability))

    def predict_seconds(self, snapshot: ResourceSnapshot, sample_count: int, batch_size: int,
                        epochs: int, upload_ratio: float, model_bytes: int) -> tuple[float, float, float]:
        batches = int(np.ceil(max(1, sample_count) / max(1, batch_size)))
        compute = epochs * batches * self.reference_batch_seconds / max(snapshot.compute_speed, 1e-9)
        communication = (upload_ratio * model_bytes * 8.0 / (max(snapshot.uplink_mbps, 1e-9) * 1_000_000.0)) + snapshot.rtt_ms / 1000.0
        return float(compute), float(communication), float(compute + communication)

    @staticmethod
    def _normalized(values: dict[int, float]) -> dict[int, float]:
        if not values:
            return {}
        low, high = min(values.values()), max(values.values())
        if high - low < 1e-12:
            return {key: 1.0 for key in values}
        return {key: (value - low) / (high - low) for key, value in values.items()}

    @staticmethod
    def _budget_utilization(privacy_states: dict[int, Any]) -> float:
        used, total = 0.0, 0.0
        for state in privacy_states.values():
            if not state:
                continue
            accountant = state["accountant"]
            used += float(accountant.epsilon)
            total += float(accountant.target_epsilon)
        if total <= 0:
            return 0.0
        return float(np.clip(used / total, 0.0, 1.0))

    def _deadline_probability(self, snapshot: ResourceSnapshot, sample_count: int, batch_size: int,
                              base_epochs: int, model_bytes: int) -> float:
        """Estimate future deadline feasibility without granting privacy spend.

        This is intentionally explainable rather than learned: it uses the best
        feasible low-cost action under the current resource snapshot as a proxy
        for how often this client may have usable future participation windows.
        """
        best_total = np.inf
        for epochs in range(int(base_epochs), 0, -1):
            for ratio in sorted(self.upload_ratios):
                _, _, total = self.predict_seconds(snapshot, sample_count, batch_size, epochs, ratio, model_bytes)
                best_total = min(best_total, total)
        if not np.isfinite(best_total) or best_total > self.deadline_seconds:
            return 0.05
        margin = (self.deadline_seconds - best_total) / max(self.deadline_seconds, 1e-9)
        return float(np.clip(0.35 + 0.65 * margin, 0.05, 1.0))

    @staticmethod
    def _expected_opportunities(remaining_rounds: int, availability: float, deadline_probability: float,
                                selection_probability: float) -> float:
        return float(max(0.25, remaining_rounds * availability * deadline_probability * selection_probability))

    def plan(self, round_num: int, sample_counts: dict[int, int], batch_size: int, base_epochs: int,
             model_bytes: int, participation_counts: list[int], contribution_scores: dict[int, float],
             quality_scores: dict[int, float], privacy_states: dict[int, Any], remaining_rounds: int,
             target_count: int, risk_actions: dict[int, str] | None = None,
             eligible_indices: set[int] | None = None) -> tuple[dict[int, PlannedAction], list[dict[str, Any]]]:
        risk_actions = risk_actions or {}
        candidates, trace = {}, []
        slack_values, debt_values = {}, {}
        budget_utilization = self._budget_utilization(privacy_states)
        total_round_estimate = max(1.0, float(round_num + remaining_rounds - 1))
        expected_budget_progress = float(np.clip(round_num / total_round_estimate, 0.0, 1.0))
        budget_lag = max(0.0, expected_budget_progress - budget_utilization)
        if self.enable_budget_utilization_boost:
            privacy_boost = float(np.clip(1.0 + self.privacy_boost_gain * budget_lag, 1.0, self.max_privacy_boost))
        else:
            privacy_boost = 1.0
        eligible_count = len(eligible_indices) if eligible_indices is not None else len(self.profiles)
        selection_probability = float(np.clip(target_count / max(1, eligible_count), 0.05, 1.0))
        opportunity_values = {}
        for idx in self.profiles:
            snapshot = self.snapshot(idx, round_num)
            deadline_prob = self._deadline_probability(
                snapshot, sample_counts[idx], batch_size, base_epochs, model_bytes
            )
            if self.enable_opportunity_privacy:
                opportunity_values[idx] = self._expected_opportunities(
                    remaining_rounds,
                    snapshot.profile.availability,
                    deadline_prob,
                    selection_probability,
                )
            else:
                opportunity_values[idx] = float(max(1, remaining_rounds))
        max_opportunities = max(opportunity_values.values()) if opportunity_values else 1.0
        for idx in self.profiles:
            snapshot = self.snapshot(idx, round_num)
            risk_action = risk_actions.get(idx, "none")
            if eligible_indices is not None and idx not in eligible_indices:
                trace.append(self._trace_row(snapshot, None, "failure_plan"))
                continue
            if not snapshot.online or risk_action == "quarantine":
                trace.append(self._trace_row(snapshot, None, "unavailable" if not snapshot.online else "quarantine"))
                continue
            state = privacy_states.get(idx)
            deadline_feasible = False
            privacy_rejected = False
            for epochs in range(int(base_epochs), 0, -1):
                chosen = None
                for ratio in self.upload_ratios:
                    comp, comm, total = self.predict_seconds(snapshot, sample_counts[idx], batch_size, epochs, ratio, model_bytes)
                    if total > self.deadline_seconds:
                        continue
                    deadline_feasible = True
                    steps = epochs * int(np.ceil(max(1, sample_counts[idx]) / max(1, batch_size)))
                    if self.enable_low_resource_compensation:
                        opportunity_compensation = float(
                            np.clip(1.0 - opportunity_values[idx] / max(max_opportunities, 1e-9), 0.0, 1.0)
                        )
                    else:
                        opportunity_compensation = 0.0
                    effective_work_ratio = float(np.clip((epochs * ratio) / max(1, base_epochs), 0.15, 1.0))
                    privacy_choice = self._privacy_choice(
                        state,
                        steps,
                        remaining_rounds,
                        quality_scores.get(idx, 0.0),
                        contribution_scores.get(idx, 0.0),
                        risk_action,
                        expected_opportunities=opportunity_values[idx],
                        budget_utilization=budget_utilization,
                        privacy_boost=privacy_boost,
                        opportunity_compensation=opportunity_compensation,
                        effective_work_ratio=effective_work_ratio,
                    )
                    noise, spend, target, cap_reason, noise_reason = privacy_choice
                    if state is not None and noise is None:
                        privacy_rejected = True
                        continue
                    chosen = PlannedAction(
                        idx,
                        epochs,
                        ratio,
                        noise,
                        comp,
                        comm,
                        total,
                        spend,
                        target,
                        opportunity_values[idx],
                        budget_utilization,
                        privacy_boost,
                        opportunity_compensation,
                        cap_reason,
                        noise_reason,
                        0.0,
                    )
                    break
                if chosen is not None:
                    candidates[idx] = chosen
                    slack_values[idx] = max(0.0, self.deadline_seconds - chosen.predicted_total_seconds)
                    debt_values[idx] = max(participation_counts) - participation_counts[idx] if participation_counts else 0.0
                    break
            if idx not in candidates:
                status = "privacy_budget_infeasible" if deadline_feasible and privacy_rejected else "deadline_infeasible"
                trace.append(self._trace_row(snapshot, None, status))

        slack = self._normalized(slack_values)
        debt = self._normalized(debt_values)
        contribution = self._normalized({idx: abs(contribution_scores.get(idx, 0.0)) for idx in candidates})
        for idx, action in list(candidates.items()):
            utility = 0.30 * quality_scores.get(idx, 0.0) + 0.25 * contribution.get(idx, 0.0) + 0.20 * slack.get(idx, 0.0) + 0.25 * debt.get(idx, 0.0)
            candidates[idx] = replace(action, utility_score=float(utility))

        selected = self._select_with_tier_coverage(candidates, target_count)
        for idx, action in candidates.items():
            snapshot = self.snapshot(idx, round_num)
            trace.append(self._trace_row(snapshot, action, "selected" if idx in selected else "not_selected"))
        return {idx: candidates[idx] for idx in selected}, trace

    def _privacy_choice(self, state: Any, steps: int, remaining_rounds: int, quality: float,
                        contribution: float, risk_action: str, expected_opportunities: float,
                        budget_utilization: float, privacy_boost: float,
                        opportunity_compensation: float, effective_work_ratio: float) -> tuple[float | None, float, float, str, str]:
        if state is None:
            return None, 0.0, 0.0, "no_client_privacy", "not_applicable"
        accountant = state["accountant"]
        remaining = accountant.remaining_epsilon
        if remaining <= 1e-12:
            return None, 0.0, 0.0, "budget_exhausted", "no_feasible_noise"
        value_signal = float(np.clip((quality + abs(contribution)) / 2.0, 0.0, 1.0))
        utility_factor = 0.75 + 0.25 * value_signal
        opportunity_factor = 1.0 + self.opportunity_compensation_weight * float(np.clip(opportunity_compensation, 0.0, 1.0))
        work_factor = float(np.sqrt(np.clip(effective_work_ratio, 0.15, 1.0)))
        target = remaining / max(1.0, expected_opportunities)
        target *= utility_factor * opportunity_factor * work_factor * privacy_boost
        cap_reason = "opportunity_adjusted"
        # A strict remaining-budget / remaining-rounds split can be lower than the
        # privacy cost of one valid DP-SGD action, causing a cold-start deadlock.
        # Reserve a one-time feasible spend; all later actions use the dynamic rule.
        if accountant.epsilon <= 1e-12:
            target = max(target, min(remaining, self.minimum_initial_privacy_increment))
            cap_reason = "cold_start_minimum"
        if risk_action in {"warning", "downweight"}:
            target *= 0.7
            cap_reason = f"risk_{risk_action}_cap"
        target = float(min(max(target, 1e-8), remaining))
        base = state["base_noise_multiplier"]
        feasible = []
        for multiplier in (0.60, 0.75, 0.85, 1.0, 1.15, 1.35, 1.60, 2.0, 2.5, 3.0, 4.0):
            noise = base * multiplier
            projected = accountant.projected_epsilon(steps, state["sample_rate"], noise)
            increment = projected - accountant.epsilon
            if projected <= accountant.target_epsilon + 1e-10:
                feasible.append((noise, increment))
                if increment <= target + 1e-10:
                    return float(noise), float(increment), target, cap_reason, "smallest_noise_within_dynamic_target"
        if budget_utilization < 0.6 and feasible and risk_action not in {"warning", "downweight"}:
            noise, increment = max(feasible, key=lambda item: item[1])
            return float(noise), float(increment), target, "utilization_catchup", "smallest_feasible_noise_under_total_budget"
        return None, 0.0, target, cap_reason, "no_feasible_noise_within_target"

    def _select_with_tier_coverage(self, candidates: dict[int, PlannedAction], target_count: int) -> list[int]:
        ranked = sorted(candidates, key=lambda idx: (candidates[idx].utility_score, -idx), reverse=True)
        limit = min(max(0, int(target_count)), len(ranked))
        if limit == 0:
            return []
        selected = []
        if self.enforce_tier_coverage and limit >= 3:
            for tier in RESOURCE_TIERS:
                tier_candidates = [idx for idx in ranked if self.profiles[idx].tier == tier]
                if tier_candidates:
                    selected.append(tier_candidates[0])
        for idx in ranked:
            if idx not in selected and len(selected) < limit:
                selected.append(idx)
        return selected[:limit]

    @staticmethod
    def _trace_row(snapshot: ResourceSnapshot, action: PlannedAction | None, status: str) -> dict[str, Any]:
        row = {
            "round": snapshot.round_num, "client_idx": snapshot.profile.client_idx, "tier": snapshot.profile.tier,
            "compute_speed": snapshot.compute_speed, "uplink_mbps": snapshot.uplink_mbps,
            "rtt_ms": snapshot.rtt_ms, "online": snapshot.online, "status": status,
        }
        if action is not None:
            row.update({"epochs": action.epochs, "upload_ratio": action.upload_ratio,
                        "noise_multiplier": action.noise_multiplier, "privacy_target_increment": action.privacy_target_increment,
                        "privacy_budget_target": action.privacy_budget_target,
                        "expected_future_opportunities": action.expected_future_opportunities,
                        "budget_utilization": action.budget_utilization,
                        "privacy_boost": action.privacy_boost,
                        "opportunity_compensation": action.opportunity_compensation,
                        "privacy_cap_reason": action.privacy_cap_reason,
                        "selected_noise_reason": action.selected_noise_reason,
                        "predicted_compute_seconds": action.predicted_compute_seconds,
                        "predicted_communication_seconds": action.predicted_communication_seconds,
                        "predicted_total_seconds": action.predicted_total_seconds, "utility_score": action.utility_score})
        return row

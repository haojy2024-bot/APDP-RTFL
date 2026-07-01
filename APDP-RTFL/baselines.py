import csv
import copy
import hashlib
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew, entropy
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from data_utils import load_experiment_data, split_data_for_clients
from fl_client import FLClient
from fl_server import FLServer
import charting as charts
from experiment_artifacts import export_tcm_checkpoints, write_artifact_manifest, write_data_artifacts
from privacy_accounting import RDPAccountant, calibrate_noise_multiplier
from resource_orchestrator import (
    ResourcePrivacyOrchestrator,
    build_resource_profiles,
    mask_delta,
    parameter_bytes,
    rotating_block_mask,
)


TOTAL_PRIVACY_BUDGET = 5.0
MIN_EPSILON = 0.1
MAX_EPSILON = 2.0
DP_EPSILON = 1.0
DP_DELTA = 1e-5
DP_L2_NORM_CLIP = 1.0
BASE_LEARNING_RATE = 0.01
EARLYSTOP_PATIENCE = 3
BASELINE_METHODS = ("dp_fedavg", "dp_fedprox", "dp_fedsgd", "dp_fednova", "grail_fl")
LEGACY_BASELINE_METHODS = ("dp_fl", "dp_flprox", "dp_rtfl", "apdp_rtfl", "dp_fedadam")
SUPPORTED_BASELINE_METHODS = ("fedavg", "fedprox", "ldp_fl") + BASELINE_METHODS + LEGACY_BASELINE_METHODS
PARTICIPATION_POLICIES = ("all", "random", "apdp_score")
PRIVACY_SENSITIVITY_METHODS = ("dp_fedavg", "dp_fedprox", "dp_fedsgd", "dp_fednova", "grail_fl")
POLLUTION_METHODS = ("grail_fl", "apdp_rtfl")
FAIRNESS_METHODS = ("dp_fedavg", "dp_fedprox", "dp_fedsgd", "dp_fednova", "grail_fl")
CONTRIBUTION_METHODS = ("dp_fedavg", "dp_fedprox", "dp_fedsgd", "dp_fednova", "grail_fl")
AUDIT_METHODS = ("dp_fedavg", "dp_fedprox", "dp_fedsgd", "dp_fednova", "grail_fl")
ABLATION_SCENARIOS = (
    "full",
    "no_adaptive_privacy",
    "no_compute_adapter",
    "no_resource_orchestration",
    "no_partial_updates",
    "no_resource_fairness",
    "no_opportunity_privacy",
    "no_budget_utilization_boost",
    "no_low_resource_compensation",
    "no_zkip",
    "no_ebcd",
    "no_tcm",
    "no_regulatory",
    "no_contribution",
    "no_fairness",
)
SYNTHETIC_FAIRNESS_DATASETS = ("emnist", "femnist", "cifar10", "cifar100")


class PrivacyRuntimeConfig:
    def __init__(self, total_budget=TOTAL_PRIVACY_BUDGET, min_epsilon=MIN_EPSILON,
                 max_epsilon=MAX_EPSILON, dp_epsilon=DP_EPSILON, dp_delta=DP_DELTA,
                 dp_l2_norm_clip=DP_L2_NORM_CLIP, failure_prob=0.15,
                 apdp_warmup_rounds=20, adaptive_increase_factor=1.10,
                 adaptive_decrease_factor=0.90, disable_compute_epoch_scaling=False,
                 epsilon_per_client_total=5.0, dp_batch_size=256):
        self.total_budget = total_budget
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_l2_norm_clip = dp_l2_norm_clip
        self.failure_prob = failure_prob
        self.apdp_warmup_rounds = apdp_warmup_rounds
        self.adaptive_increase_factor = adaptive_increase_factor
        self.adaptive_decrease_factor = adaptive_decrease_factor
        self.disable_compute_epoch_scaling = disable_compute_epoch_scaling
        self.epsilon_per_client_total = float(epsilon_per_client_total)
        self.dp_batch_size = int(dp_batch_size)
        self.budget_semantics = "per_client_total"


def make_privacy_config(args):
    epsilon_per_client_total = (
        args.epsilon_per_client_total
        if getattr(args, "epsilon_per_client_total", None) is not None
        else args.total_privacy_budget
    )
    return PrivacyRuntimeConfig(
        total_budget=args.total_privacy_budget,
        min_epsilon=args.min_epsilon,
        max_epsilon=args.max_epsilon,
        dp_epsilon=args.dp_epsilon,
        dp_delta=args.dp_delta,
        dp_l2_norm_clip=args.dp_l2_norm_clip,
        failure_prob=args.failure_prob,
        apdp_warmup_rounds=args.apdp_warmup_rounds,
        adaptive_increase_factor=args.adaptive_increase_factor,
        adaptive_decrease_factor=args.adaptive_decrease_factor,
        disable_compute_epoch_scaling=args.disable_compute_epoch_scaling,
        epsilon_per_client_total=epsilon_per_client_total,
        dp_batch_size=args.dp_batch_size,
    )


METHOD_CONFIGS = {
    "fedavg": {
        "label": "FedAvg",
        "reference": "",
        "dp_scope": "none",
        "use_dp": False,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": False,
        "use_ebcd": False,
        "use_tcm": False,
        "fedprox": False,
        "force_client_epochs": None,
        "resource_orchestrator": False,
    },
    "fedprox": {
        "label": "FedProx",
        "reference": "Li et al., Federated optimization in heterogeneous networks, MLSys 2020.",
        "dp_scope": "none",
        "use_dp": False,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": False,
        "use_ebcd": False,
        "use_tcm": False,
        "fedprox": True,
        "force_client_epochs": None,
        "resource_orchestrator": False,
    },
    "ldp_fl": {
        "label": "LDP-FL",
        "reference": "Arachchige et al., Local differential privacy for deep learning, IEEE Internet of Things Journal 2019/2020.",
        "dp_scope": "client",
        "use_dp": True,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": False,
        "use_ebcd": False,
        "use_tcm": False,
        "fedprox": False,
        "force_client_epochs": None,
        "resource_orchestrator": False,
    },
    "dp_fl": {
        "label": "DP-FedAvg",
        "reference": "Arachchige et al., Local differential privacy for deep learning, IEEE Internet of Things Journal 2019/2020.",
        "dp_scope": "client",
        "use_dp": True,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": False,
        "use_ebcd": False,
        "use_tcm": False,
        "fedprox": False,
        "force_client_epochs": None,
        "resource_orchestrator": False,
    },
    "dp_fedavg": {
        "label": "DP-FedAvg",
        "reference": "McMahan et al., Learning Differentially Private Recurrent Language Models, ICLR 2018; McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data, AISTATS 2017.",
        "dp_scope": "client",
        "use_dp": True,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": False,
        "use_ebcd": False,
        "use_tcm": False,
        "fedprox": False,
        "force_client_epochs": None,
        "resource_orchestrator": False,
    },
    "dp_flprox": {
        "label": "DP-FedProx",
        "reference": "Li et al., Federated optimization in heterogeneous networks, MLSys 2020.",
        "dp_scope": "client",
        "use_dp": True,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": False,
        "use_ebcd": False,
        "use_tcm": False,
        "fedprox": True,
        "force_client_epochs": None,
        "resource_orchestrator": False,
    },
    "dp_fedprox": {
        "label": "DP-FedProx",
        "reference": "Li et al., Federated optimization in heterogeneous networks, MLSys 2020.",
        "dp_scope": "client",
        "use_dp": True,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": False,
        "use_ebcd": False,
        "use_tcm": False,
        "fedprox": True,
        "force_client_epochs": None,
        "resource_orchestrator": False,
    },
    "dp_fedsgd": {
        "label": "DP-FedSGD",
        "reference": "Auddy et al., Statistical Limits and Efficient Algorithms for Differentially Private Federated Learning, arXiv:2605.18656, 2026.",
        "dp_scope": "client",
        "use_dp": True,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": False,
        "use_ebcd": False,
        "use_tcm": False,
        "fedprox": False,
        "force_client_epochs": 1,
        "resource_orchestrator": False,
    },
    "dp_fednova": {
        "label": "DP-FedNova",
        "reference": "Wang et al., Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization, NeurIPS 2020.",
        "dp_scope": "client",
        "use_dp": True,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": False,
        "use_ebcd": False,
        "use_tcm": False,
        "fedprox": False,
        "force_client_epochs": None,
        "resource_orchestrator": False,
        "aggregation": "fednova",
    },
    "dp_fedadam": {
        "label": "DP-FedAdam",
        "reference": "Reddi et al., Adaptive Federated Optimization, ICLR 2021.",
        "dp_scope": "client",
        "use_dp": True,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": False,
        "use_ebcd": False,
        "use_tcm": False,
        "fedprox": False,
        "force_client_epochs": None,
        "resource_orchestrator": False,
        "aggregation": "fedadam",
        "server_learning_rate": 0.01,
        "server_beta1": 0.9,
        "server_beta2": 0.99,
        "server_tau": 1e-3,
    },
    "global_dp": {
        "label": "Global-DP",
        "reference": "",
        "dp_scope": "server",
        "use_dp": False,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": False,
        "use_ebcd": False,
        "use_tcm": False,
        "fedprox": False,
        "force_client_epochs": None,
        "resource_orchestrator": False,
    },
    "dp_rtfl": {
        "label": "DP-RTFL",
        "reference": "",
        "dp_scope": "client",
        "use_dp": True,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": True,
        "use_ebcd": True,
        "use_tcm": True,
        "fedprox": False,
        "force_client_epochs": None,
        "resource_orchestrator": False,
    },
    "apdp_rtfl": {
        "label": "GRAIL-FL",
        "reference": "",
        "dp_scope": "client",
        "use_dp": True,
        "dynamic_privacy": True,
        "compute_adapter": True,
        "use_zkip": True,
        "use_ebcd": True,
        "use_tcm": True,
        "fedprox": False,
        "force_client_epochs": None,
        "resource_orchestrator": True,
    },
    "grail_fl": {
        "label": "GRAIL-FL",
        "reference": "",
        "dp_scope": "client",
        "use_dp": True,
        "dynamic_privacy": True,
        "compute_adapter": True,
        "use_zkip": True,
        "use_ebcd": True,
        "use_tcm": True,
        "fedprox": False,
        "force_client_epochs": None,
        "resource_orchestrator": True,
    },
}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _metric_value(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _compute_data_quality_score(y_data):
    if y_data is None or len(y_data) == 0:
        return 0.0
    _, counts = np.unique(y_data, return_counts=True)
    probabilities = counts / len(y_data)
    data_entropy = entropy(probabilities)
    max_entropy = np.log(len(counts)) if len(counts) > 0 else 1.0
    return data_entropy / max_entropy if max_entropy > 0 else 0.0


class SyntheticSensitiveAttributeAssigner:
    def __init__(self, attrs="gender,age,region", profile="regulated"):
        self.attrs = {item.strip().lower() for item in attrs.split(",") if item.strip()}
        self.profile = profile

    def assign(self, client_datasets):
        rows = []
        for idx, (_, y_data) in enumerate(client_datasets):
            region_group = ("east", "central", "west")[idx % 3]
            age_group = ("young", "middle", "old")[(idx // 2) % 3]
            gender_group = "A" if idx % 2 == 0 else "B"
            compute_level = {"east": "high", "central": "medium", "west": "low"}[region_group]
            if age_group == "old" or region_group == "west":
                data_quality = "noisy"
            elif gender_group == "B":
                data_quality = "biased"
            else:
                data_quality = "clean"
            labels = np.asarray(y_data).astype(int)
            rows.append(
                {
                    "client_idx": idx,
                    "client_id": f"client_{idx}",
                    "gender_group": gender_group if "gender" in self.attrs else "unspecified",
                    "age_group": age_group if "age" in self.attrs else "unspecified",
                    "region_group": region_group if "region" in self.attrs else "unspecified",
                    "compute_level": compute_level,
                    "data_quality": data_quality,
                    "sample_count": len(labels),
                    "class_coverage": len(np.unique(labels)) if len(labels) else 0,
                }
            )
        return rows


class SyntheticFairnessPressureApplier:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def _filter_by_gender(self, X_data, y_data, metadata):
        if metadata["gender_group"] != "B" or len(y_data) <= 2:
            return X_data, y_data
        labels = np.asarray(y_data).astype(int)
        unique = np.unique(labels)
        keep_classes = set(unique[:max(1, int(np.ceil(len(unique) * 0.7)))].tolist())
        keep_mask = np.array([label in keep_classes for label in labels])
        if keep_mask.sum() < max(2, int(len(labels) * 0.3)):
            return X_data, y_data
        return X_data[keep_mask], y_data[keep_mask]

    def _filter_by_region(self, X_data, y_data, metadata):
        region = metadata["region_group"]
        if region == "central" or len(y_data) <= 2:
            return X_data, y_data
        labels = np.asarray(y_data).astype(int)
        unique = np.unique(labels)
        midpoint = max(1, len(unique) // 2)
        preferred = set(unique[:midpoint].tolist()) if region == "east" else set(unique[midpoint:].tolist())
        preferred_mask = np.array([label in preferred for label in labels])
        keep_prob = np.where(preferred_mask, 1.0, 0.35)
        keep_mask = self.rng.random(len(labels)) < keep_prob
        if keep_mask.sum() < max(2, int(len(labels) * 0.25)):
            return X_data, y_data
        return X_data[keep_mask], y_data[keep_mask]

    def _apply_quality_noise(self, X_data, metadata):
        quality = metadata["data_quality"]
        age = metadata["age_group"]
        if quality == "clean" and age == "young":
            return X_data
        X_mod = np.copy(X_data)
        noise_std = 0.08 if age == "middle" else 0.18
        if quality == "noisy":
            noise_std += 0.12
        X_mod = X_mod + self.rng.normal(0, noise_std, size=X_mod.shape)
        dropout_prob = 0.03 if quality == "biased" else 0.0
        if age == "old":
            dropout_prob += 0.07
        if dropout_prob > 0:
            mask = self.rng.random(X_mod.shape) < dropout_prob
            X_mod[mask] = 0.0
        return X_mod

    def apply(self, client_datasets, metadata_rows):
        pressured = []
        updated_metadata = []
        for X_data, y_data in client_datasets:
            X_arr = np.asarray(X_data, dtype=float)
            y_arr = np.asarray(y_data).astype(int)
            metadata = metadata_rows[len(pressured)].copy()
            X_arr, y_arr = self._filter_by_gender(X_arr, y_arr, metadata)
            X_arr, y_arr = self._filter_by_region(X_arr, y_arr, metadata)
            X_arr = self._apply_quality_noise(X_arr, metadata)
            if len(y_arr) == 0:
                X_arr = np.asarray(X_data[:1], dtype=float)
                y_arr = np.asarray(y_data[:1]).astype(int)
            metadata["sample_count"] = len(y_arr)
            metadata["class_coverage"] = len(np.unique(y_arr)) if len(y_arr) else 0
            updated_metadata.append(metadata)
            pressured.append((X_arr, y_arr))
        return pressured, updated_metadata


class GroupFairnessEvaluator:
    ATTRIBUTES = ("gender_group", "age_group", "region_group")

    @staticmethod
    def _client_index(client_id):
        return int(str(client_id).split("_")[-1])

    def __init__(self, metadata_rows):
        self.metadata_by_client = {row["client_id"]: row for row in metadata_rows}

    def group_rows(self, dataset_name, method_name, result):
        rows = []
        fairness_records = result.get("fairness_records", [])
        regulatory_rows = result.get("regulatory_records", [])
        for attr in self.ATTRIBUTES:
            groups = sorted({row[attr] for row in self.metadata_by_client.values()})
            for round_num in result.get("rounds", []):
                group_metrics = []
                for group in groups:
                    clients = [cid for cid, meta in self.metadata_by_client.items() if meta[attr] == group]
                    perf_rows = [
                        row for row in fairness_records
                        if row["round"] == round_num and row["client_id"] in clients
                    ]
                    acc_values = [row["local_accuracy"] for row in perf_rows if np.isfinite(_metric_value(row["local_accuracy"]))]
                    f1_values = [row["local_f1_score"] for row in perf_rows if np.isfinite(_metric_value(row["local_f1_score"]))]
                    bal_values = [row["local_balanced_accuracy"] for row in perf_rows if np.isfinite(_metric_value(row["local_balanced_accuracy"]))]
                    auc_values = [row["local_auc_roc"] for row in perf_rows if np.isfinite(_metric_value(row.get("local_auc_roc")))]
                    avg_acc = float(np.mean(acc_values)) if acc_values else np.nan
                    avg_f1 = float(np.mean(f1_values)) if f1_values else np.nan
                    avg_bal = float(np.mean(bal_values)) if bal_values else np.nan
                    avg_auc = float(np.mean(auc_values)) if auc_values else np.nan
                    group_metrics.append((group, avg_acc, avg_f1))
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "method": method_name,
                            "round": round_num,
                            "attribute": attr,
                            "group": group,
                            "accuracy": avg_acc,
                            "macro_f1": avg_f1,
                            "balanced_accuracy": avg_bal,
                            "auc": avg_auc,
                            "equal_opportunity": np.nan,
                        }
                    )
                acc_values = [item[1] for item in group_metrics if np.isfinite(_metric_value(item[1]))]
                f1_values = [item[2] for item in group_metrics if np.isfinite(_metric_value(item[2]))]
                if acc_values:
                    gap = float(max(acc_values) - min(acc_values))
                    worst = float(min(acc_values))
                else:
                    gap = 0.0
                    worst = np.nan
                f1_gap = float(max(f1_values) - min(f1_values)) if f1_values else 0.0
                rows.append(
                    {
                        "dataset": dataset_name,
                        "method": method_name,
                        "round": round_num,
                        "attribute": attr,
                        "group": "__summary__",
                        "accuracy": np.nan,
                        "macro_f1": np.nan,
                        "balanced_accuracy": np.nan,
                        "auc": np.nan,
                        "equal_opportunity": np.nan,
                        "worst_group_accuracy": worst,
                        "group_accuracy_gap": gap,
                        "group_f1_gap": f1_gap,
                    }
                )
        return rows

    def federated_rows(self, dataset_name, method_name, result):
        rows = []
        regulatory_rows = result.get("regulatory_records", [])
        fairness_records = result.get("fairness_records", [])
        for attr in self.ATTRIBUTES:
            groups = sorted({row[attr] for row in self.metadata_by_client.values()})
            for round_num in result.get("rounds", []):
                for group in groups:
                    clients = [cid for cid, meta in self.metadata_by_client.items() if meta[attr] == group]
                    fair_rows = [
                        row for row in fairness_records
                        if row["round"] == round_num and row["client_id"] in clients
                    ]
                    reg_rows = [
                        row for row in regulatory_rows
                        if row["round"] == round_num and row["client_id"] in clients
                    ]
                    eps = [row["epsilon"] for row in fair_rows if row["epsilon"] is not None and np.isfinite(_metric_value(row["epsilon"]))]
                    participation = [row["participation_count"] for row in fair_rows]
                    adjusted_weights = [row["adjusted_weight"] for row in reg_rows if np.isfinite(_metric_value(row["adjusted_weight"]))]
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "method": method_name,
                            "round": round_num,
                            "attribute": attr,
                            "group": group,
                            "avg_epsilon": float(np.mean(eps)) if eps else np.nan,
                            "avg_aggregation_weight": float(np.mean(adjusted_weights)) if adjusted_weights else np.nan,
                            "avg_participation_rounds": float(np.mean(participation)) if participation else 0.0,
                            "warning_count": sum(1 for row in reg_rows if row["action"] == "warning"),
                            "downweight_count": sum(1 for row in reg_rows if row["action"] == "downweight"),
                            "quarantine_count": sum(1 for row in reg_rows if row["action"] == "quarantine"),
                        }
                    )
        return rows


def _assign_capabilities(num_clients):
    default_dist = [1.0, 0.85, 0.65, 0.40, 0.25][:num_clients]
    if len(default_dist) < num_clients:
        default_dist += [0.5] * (num_clients - len(default_dist))
    return {i: default_dist[i] for i in range(num_clients)}


def _effective_epochs(client_idx, base_epochs, capabilities, enabled, privacy_config):
    if not enabled or privacy_config.disable_compute_epoch_scaling:
        return base_epochs
    cap = capabilities.get(client_idx, 1.0)
    return max(1, int(base_epochs * cap * 1.1))


def _adjust_epsilon_for_compute(base_epsilon, client_idx, capabilities, enabled, privacy_config):
    if not enabled:
        return base_epsilon
    cap = capabilities.get(client_idx, 1.0)
    if cap < 0.3:
        return privacy_config.max_epsilon
    adjustment = 1.0 + (1.0 - cap) * 0.8
    return min(privacy_config.max_epsilon, max(privacy_config.min_epsilon, base_epsilon * adjustment))


def _allocate_budget(clients, active_indices, capabilities, dynamic, privacy_config):
    # Kept for legacy visualisations only. Client-DP budgets are independent,
    # never divided between clients, and enforced by per-client RDP ledgers.
    return {idx: privacy_config.epsilon_per_client_total for idx in active_indices}


def _adaptive_adjust(clients, previous_allocations, round_num, privacy_config):
    return previous_allocations.copy()


def _make_client_privacy_state(clients, args, privacy_config, config):
    state = {}
    for idx, client in enumerate(clients):
        n_samples = max(1, len(client.y_train))
        batch_size = min(privacy_config.dp_batch_size, n_samples)
        sample_rate = batch_size / n_samples
        epochs = int(config.get("force_client_epochs") or args.client_epochs)
        planned_steps = args.num_rounds * epochs * int(np.ceil(n_samples / batch_size))
        base_sigma = calibrate_noise_multiplier(
            sample_rate, planned_steps, privacy_config.epsilon_per_client_total, privacy_config.dp_delta
        )
        # APDP may lower noise by the configured increase factor after warmup;
        # reserve that possible spend up front so it cannot overshoot the ledger.
        if config.get("dynamic_privacy"):
            base_sigma *= privacy_config.adaptive_increase_factor
        state[idx] = {
            "accountant": RDPAccountant(privacy_config.epsilon_per_client_total, privacy_config.dp_delta),
            "base_noise_multiplier": base_sigma,
            "sample_rate": sample_rate,
        }
    return state


def _apdp_noise_multiplier(idx, capabilities, participation_counts, state, privacy_config, round_num):
    base = state["base_noise_multiplier"]
    if round_num <= privacy_config.apdp_warmup_rounds:
        return base
    median_capability = float(np.median(list(capabilities.values()))) if capabilities else 1.0
    median_participation = float(np.median(participation_counts)) if participation_counts else 0.0
    rewarded = capabilities.get(idx, 1.0) >= median_capability or participation_counts[idx] <= median_participation
    factor = privacy_config.adaptive_increase_factor if rewarded else privacy_config.adaptive_decrease_factor
    return base / factor


def _split_train_val(client_datasets, classes, seed):
    train_val = []
    for i, (X_c, y_c) in enumerate(client_datasets):
        if X_c.shape[0] > 5 and len(np.unique(y_c)) >= 2:
            counts = np.bincount(y_c.astype(int), minlength=len(classes))
            present_classes = np.count_nonzero(counts)
            val_size = int(np.ceil(X_c.shape[0] * 0.2))
            train_size = X_c.shape[0] - val_size
            stratify = (
                y_c
                if np.min(counts[counts > 0]) >= 2
                and val_size >= present_classes
                and train_size >= present_classes
                else None
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_c, y_c, test_size=0.2, random_state=seed + i, stratify=stratify
            )
        else:
            X_train, y_train = X_c, y_c
            X_val, y_val = None, None
        train_val.append((X_train, y_train, X_val, y_val))
    return train_val


def _init_clients(train_val_data, num_features, classes, seed, privacy_config):
    clients = []
    for i, (X_train, y_train, X_val, y_val) in enumerate(train_val_data):
        clients.append(
            FLClient(
                f"client_{i}",
                X_train,
                y_train,
                num_features,
                learning_rate=BASE_LEARNING_RATE,
                dp_epsilon=privacy_config.dp_epsilon,
                dp_delta=privacy_config.dp_delta,
                dp_l2_norm_clip=privacy_config.dp_l2_norm_clip,
                random_state=seed + i,
                X_val=X_val,
                y_val=y_val,
                earlystop_patience=EARLYSTOP_PATIENCE,
                classes=classes,
                dp_batch_size=privacy_config.dp_batch_size,
            )
        )
    return clients


def _init_backend_clients(train_val_data, num_features, classes, args, privacy_config):
    backend = getattr(args, "backend", "sklearn")
    if backend == "sklearn":
        return _init_clients(train_val_data, num_features, classes, args.seed, privacy_config), "sklearn", None
    if backend != "torch":
        raise ValueError(f"Unsupported backend: {backend}")
    try:
        from torch_baselines import TorchLinearClient, _parse_mlp_hidden, _resolve_device
    except ImportError as exc:
        raise RuntimeError(
            "The torch backend requires PyTorch. Install a CUDA-enabled PyTorch build "
            "on the experiment server, or rerun with --backend sklearn."
        ) from exc

    device = _resolve_device(args.device)
    clients = []
    for i, (X_train, y_train, X_val, y_val) in enumerate(train_val_data):
        clients.append(
            TorchLinearClient(
                f"client_{i}",
                X_train,
                y_train,
                num_features,
                classes,
                device,
                X_val=X_val,
                y_val=y_val,
                learning_rate=BASE_LEARNING_RATE,
                batch_size=args.torch_batch_size,
                random_state=args.seed + i,
                privacy_config=privacy_config,
                model_type=getattr(args, "torch_model", "linear"),
                mlp_hidden=_parse_mlp_hidden(getattr(args, "torch_mlp_hidden", "256,128")),
            )
        )
    return clients, "torch", device


def _dp_noise_stddev(privacy_config, epsilon=None):
    effective_epsilon = privacy_config.dp_epsilon if epsilon is None else epsilon
    if effective_epsilon <= 0:
        return 0.0
    return (
        privacy_config.dp_l2_norm_clip
        * np.sqrt(2 * np.log(1.25 / privacy_config.dp_delta))
        / effective_epsilon
    )


def _apply_server_dp_to_delta(delta, privacy_config):
    total_norm = np.sqrt(sum(np.linalg.norm(v.flatten()) ** 2 for v in delta.values()))
    clip_factor = min(1.0, privacy_config.dp_l2_norm_clip / (total_norm + 1e-6))
    noise_stddev = _dp_noise_stddev(privacy_config, epsilon=privacy_config.total_budget)
    noisy_delta = {}
    for key, value in delta.items():
        clipped = value * clip_factor
        noisy_delta[key] = clipped + np.random.normal(0, noise_stddev, size=value.shape)
    return noisy_delta, noise_stddev


def _local_step_count(data_size, epochs, batch_size):
    return max(1, int(epochs) * int(np.ceil(max(1, data_size) / max(1, batch_size))))


def _apply_server_optimizer(aggregated_delta, optimizer_state, config):
    if optimizer_state is None or config.get("aggregation") != "fedadam":
        return aggregated_delta
    beta1 = float(config.get("server_beta1", 0.9))
    beta2 = float(config.get("server_beta2", 0.99))
    tau = float(config.get("server_tau", 1e-3))
    server_lr = float(config.get("server_learning_rate", 0.01))
    optimizer_state["t"] = int(optimizer_state.get("t", 0)) + 1
    if "m" not in optimizer_state:
        optimizer_state["m"] = {key: np.zeros_like(value, dtype=float) for key, value in aggregated_delta.items()}
        optimizer_state["v"] = {key: np.zeros_like(value, dtype=float) for key, value in aggregated_delta.items()}
    update = {}
    for key, delta in aggregated_delta.items():
        optimizer_state["m"][key] = beta1 * optimizer_state["m"][key] + (1.0 - beta1) * delta
        optimizer_state["v"][key] = beta2 * optimizer_state["v"][key] + (1.0 - beta2) * np.square(delta)
        m_hat = optimizer_state["m"][key] / (1.0 - beta1 ** optimizer_state["t"])
        v_hat = optimizer_state["v"][key] / (1.0 - beta2 ** optimizer_state["t"])
        update[key] = server_lr * m_hat / (np.sqrt(v_hat) + tau)
    return update


def _aggregate_deltas(base_params, deltas_with_sizes, verify_zkip, zkip, privacy_config=None,
                      apply_server_dp=False, config=None, optimizer_state=None):
    config = config or {}
    valid = []
    total_weight = 0.0
    weighted_local_steps = 0.0
    aggregated_from = []
    zkip_failures = 0
    for update in deltas_with_sizes:
        local_steps = 1
        if len(update) == 6:
            delta, proof, client_id, data_size, adjusted_weight, local_steps = update
        elif len(update) == 5:
            delta, proof, client_id, data_size, adjusted_weight = update
        else:
            delta, proof, client_id, data_size = update
            adjusted_weight = data_size
        if delta is None:
            continue
        if verify_zkip and not zkip.verify_proof(delta, proof):
            zkip_failures += 1
            continue
        if adjusted_weight <= 0:
            continue
        local_steps = max(1, int(local_steps))
        valid.append((delta, adjusted_weight, local_steps))
        total_weight += adjusted_weight
        weighted_local_steps += adjusted_weight * local_steps
        aggregated_from.append(client_id)
    if not valid or total_weight == 0:
        return base_params, False, aggregated_from, zkip_failures, 0.0

    aggregated_delta = {k: np.zeros_like(v) for k, v in valid[0][0].items()}
    use_fednova = config.get("aggregation") == "fednova"
    average_local_steps = weighted_local_steps / total_weight if total_weight else 1.0
    for delta, adjusted_weight, local_steps in valid:
        weight = adjusted_weight / total_weight
        for key in aggregated_delta:
            contribution = delta[key] / local_steps if use_fednova else delta[key]
            aggregated_delta[key] += contribution * weight
    if use_fednova:
        for key in aggregated_delta:
            aggregated_delta[key] *= average_local_steps
    next_params = {k: np.copy(v) for k, v in base_params.items()}
    server_noise_scale = 0.0
    if apply_server_dp and privacy_config is not None:
        aggregated_delta, server_noise_scale = _apply_server_dp_to_delta(aggregated_delta, privacy_config)
    aggregated_delta = _apply_server_optimizer(aggregated_delta, optimizer_state, config)
    for key in aggregated_delta:
        next_params[key] += aggregated_delta[key]
    return next_params, True, aggregated_from, zkip_failures, server_noise_scale


def _aggregate_masked_deltas(base_params, deltas_with_sizes, masks_by_client, verify_zkip, zkip,
                             privacy_config=None, apply_server_dp=False):
    """Coordinate-wise aggregation for ARPA partial parameter-block uploads."""
    valid, aggregated_from, zkip_failures = [], [], 0
    for update in deltas_with_sizes:
        if len(update) == 6:
            delta, proof, client_id, data_size, adjusted_weight, _local_steps = update
        elif len(update) == 5:
            delta, proof, client_id, data_size, adjusted_weight = update
        else:
            delta, proof, client_id, data_size = update
            adjusted_weight = data_size
        if delta is None or adjusted_weight <= 0:
            continue
        if verify_zkip and not zkip.verify_proof(delta, proof):
            zkip_failures += 1
            continue
        valid.append((delta, float(adjusted_weight), masks_by_client.get(client_id)))
        aggregated_from.append(client_id)
    if not valid:
        return base_params, False, aggregated_from, zkip_failures, 0.0

    aggregated_delta = {key: np.zeros_like(value) for key, value in base_params.items()}
    for key in aggregated_delta:
        numerator = np.zeros_like(aggregated_delta[key], dtype=float)
        denominator = np.zeros_like(aggregated_delta[key], dtype=float)
        for delta, weight, masks in valid:
            mask = masks[key] if masks is not None else np.ones_like(delta[key], dtype=bool)
            numerator += np.where(mask, delta[key], 0.0) * weight
            denominator += np.where(mask, weight, 0.0)
        aggregated_delta[key] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    next_params = {key: np.copy(value) for key, value in base_params.items()}
    server_noise_scale = 0.0
    if apply_server_dp and privacy_config is not None:
        aggregated_delta, server_noise_scale = _apply_server_dp_to_delta(aggregated_delta, privacy_config)
    for key, delta in aggregated_delta.items():
        next_params[key] += delta
    return next_params, True, aggregated_from, zkip_failures, server_noise_scale


def _parse_upload_ratios(value):
    ratios = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not ratios or any(item <= 0 or item > 1 for item in ratios):
        raise ValueError("--upload-ratios must contain values in (0, 1]")
    return ratios


class RegulatoryInterventionController:
    def __init__(self, warning_threshold=1.5, quarantine_threshold=2.5, penalty_weight=0.5):
        self.warning_threshold = warning_threshold
        self.quarantine_threshold = quarantine_threshold
        self.penalty_weight = penalty_weight
        self.downweight_threshold = (warning_threshold + quarantine_threshold) / 2.0

    @staticmethod
    def _safe_float(value):
        if value is None:
            return np.nan
        try:
            value = float(value)
        except (TypeError, ValueError):
            return np.nan
        return value if np.isfinite(value) else np.nan

    def _relative_deviation(self, value, baseline):
        value = self._safe_float(value)
        baseline = self._safe_float(baseline)
        if not np.isfinite(value):
            return 0.0
        if not np.isfinite(baseline) or abs(baseline) < 1e-12:
            return abs(value)
        return abs(value - baseline) / abs(baseline)

    def _risk_score(self, update_norm, ebcd_stats, zkip_status, median_norm, median_stats):
        if zkip_status is False:
            return self.quarantine_threshold + 1.0
        norm_score = self._relative_deviation(update_norm, median_norm)
        stat_scores = []
        for idx, value in enumerate(ebcd_stats or (None, None, None)):
            stat_scores.append(self._relative_deviation(value, median_stats[idx]))
        ebcd_score = max(stat_scores) if stat_scores else 0.0
        return norm_score + 0.5 * ebcd_score

    def _action_for_score(self, score):
        if score >= self.quarantine_threshold:
            return "quarantine"
        if score >= self.downweight_threshold:
            return "downweight"
        if score >= self.warning_threshold:
            return "warning"
        return "normal"

    def evaluate_round(self, round_num, clients, selected_indices, update_norms, ebcd_stats, zkip_status):
        valid_norms = [self._safe_float(update_norms[idx]) for idx in selected_indices if np.isfinite(self._safe_float(update_norms[idx]))]
        median_norm = float(np.median(valid_norms)) if valid_norms else 0.0
        median_stats = []
        for stat_idx in range(3):
            values = [
                self._safe_float(ebcd_stats[idx][stat_idx])
                for idx in selected_indices
                if ebcd_stats[idx] is not None and np.isfinite(self._safe_float(ebcd_stats[idx][stat_idx]))
            ]
            median_stats.append(float(np.median(values)) if values else 0.0)

        records = []
        adjusted_weights = {}
        for idx in selected_indices:
            client = clients[idx]
            original_weight = len(client.y_train)
            score = self._risk_score(update_norms[idx], ebcd_stats[idx], zkip_status[idx], median_norm, median_stats)
            action = self._action_for_score(score)
            if action == "quarantine":
                adjusted_weight = 0.0
            elif action == "downweight":
                normalized_score = min(1.0, score / max(self.quarantine_threshold, 1e-12))
                adjusted_weight = original_weight * max(0.0, 1.0 - self.penalty_weight * normalized_score)
            else:
                adjusted_weight = float(original_weight)
            adjusted_weights[client.client_id] = adjusted_weight
            stat_values = ebcd_stats[idx] or (None, None, None)
            records.append(
                {
                    "round": round_num,
                    "client_id": client.client_id,
                    "risk_score": score,
                    "action": action,
                    "original_weight": original_weight,
                    "adjusted_weight": adjusted_weight,
                    "zkip_status": zkip_status[idx],
                    "update_norm": update_norms[idx],
                    "ebcd_variance": stat_values[0],
                    "ebcd_kurtosis": stat_values[1],
                    "ebcd_skewness": stat_values[2],
                }
            )
        return records, adjusted_weights


def _make_eval_server(method_name, client_ids, num_features, classes, params):
    server = FLServer(f"{method_name}_server", client_ids, num_features, classes=classes)
    server.global_model_parameters = {k: np.copy(v) for k, v in params.items()}
    return server


def _flatten_parameter_dict(params):
    arrays = []
    for key in sorted(params):
        value = np.asarray(params[key])
        arrays.append(value.reshape(-1))
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float32)


def _predict_logits_from_torch_params(params, X_eval):
    if "coef_" in params and "intercept_" in params:
        logits = np.asarray(X_eval, dtype=np.float32) @ np.asarray(params["coef_"], dtype=np.float32).T
        return logits + np.asarray(params["intercept_"], dtype=np.float32)
    activations = np.asarray(X_eval, dtype=np.float32)
    layer_indices = sorted(
        int(key.split(".")[0])
        for key in params
        if key.endswith(".weight") and key.split(".")[0].isdigit()
    )
    linear_layers = [idx for idx in layer_indices if f"{idx}.bias" in params]
    if not linear_layers:
        raise ValueError("Torch parameter dictionary does not contain linear or MLP weights.")
    for layer_pos, idx in enumerate(linear_layers):
        activations = activations @ np.asarray(params[f"{idx}.weight"], dtype=np.float32).T
        activations = activations + np.asarray(params[f"{idx}.bias"], dtype=np.float32)
        if layer_pos < len(linear_layers) - 1:
            activations = np.maximum(activations, 0.0)
    return activations


def _evaluate_softmax_params(params, X_eval, y_eval, classes):
    if X_eval is None or y_eval is None or len(y_eval) == 0:
        return {}
    class_values = np.asarray(classes, dtype=int)
    logits = _predict_logits_from_torch_params(params, X_eval)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probas = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    predictions = class_values[np.argmax(probas, axis=1)]
    average = "binary" if len(class_values) <= 2 else "macro"
    y_classes = np.unique(y_eval)
    pred_classes = np.unique(predictions)
    if len(y_classes) < 2 or len(pred_classes) < 2:
        f1 = np.nan
        precision = np.nan
        recall = np.nan
        auc_roc = np.nan
    else:
        f1 = f1_score(y_eval, predictions, average=average, zero_division=0)
        precision = precision_score(y_eval, predictions, average=average, zero_division=0)
        recall = recall_score(y_eval, predictions, average=average, zero_division=0)
        try:
            if len(class_values) <= 2:
                positive_index = 1 if probas.shape[1] > 1 else 0
                auc_roc = roc_auc_score(y_eval, probas[:, positive_index])
            elif len(y_classes) < len(class_values):
                auc_roc = np.nan
            else:
                auc_roc = roc_auc_score(y_eval, probas, multi_class="ovr", average="macro", labels=class_values)
        except ValueError:
            auc_roc = np.nan
    return {
        "accuracy": accuracy_score(y_eval, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_eval, predictions),
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc_roc,
    }


def _evaluate_global_params(method_name, client_ids, num_features, classes, params, X_test, y_test, round_num, backend):
    if backend == "torch":
        metrics = _evaluate_softmax_params(params, X_test, y_test, classes)
        pred_classes = np.unique(
            np.asarray(classes, dtype=int)[
                np.argmax(
                    _predict_logits_from_torch_params(params, X_test),
                    axis=1,
                )
            ]
        ) if X_test.shape[0] else []
        print(f"[Round {round_num}] y_test classes: {np.unique(y_test)}, predictions classes: {pred_classes}")
        return metrics
    eval_server = _make_eval_server(method_name, client_ids, num_features, classes, params)
    return eval_server.evaluate_global_model(X_test, y_test, round_num)


def _evaluate_params_on_client(params, client, classes, num_features):
    X_eval = client.X_val if client.X_val is not None and client.X_val.shape[0] > 0 else None
    y_eval = client.y_val if client.y_val is not None and len(client.y_val) > 0 else None
    if X_eval is None or y_eval is None or len(np.unique(y_eval)) < 2:
        return None
    class_values = np.asarray(classes, dtype=int)
    if "coef_" not in params or np.asarray(params["coef_"]).shape[0] == len(class_values):
        try:
            metrics = _evaluate_softmax_params(params, X_eval, y_eval, classes)
            return {
                "validation_size": len(y_eval),
                "local_accuracy": metrics.get("accuracy", np.nan),
                "local_balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
                "local_f1_score": metrics.get("f1_score", np.nan),
                "local_auc_roc": metrics.get("auc_roc", np.nan),
            }
        except Exception:
            return None
    param_classes = 1 if len(class_values) <= 2 else len(class_values)
    model = SGDClassifier(loss="log_loss")
    model.coef_ = np.zeros((param_classes, num_features))
    model.intercept_ = np.zeros(param_classes)
    model.partial_fit(X_eval[:1], y_eval[:1], classes=class_values)
    model.coef_ = np.copy(params["coef_"]).reshape(param_classes, num_features)
    model.intercept_ = np.copy(params["intercept_"]).reshape(param_classes)
    try:
        predictions = model.predict(X_eval)
        auc_roc = np.nan
        if hasattr(model, "predict_proba") and len(np.unique(y_eval)) >= 2:
            probas = model.predict_proba(X_eval)
            try:
                if len(class_values) <= 2:
                    auc_roc = roc_auc_score(y_eval, probas[:, 1])
                elif len(np.unique(y_eval)) == len(class_values):
                    auc_roc = roc_auc_score(y_eval, probas, multi_class="ovr", average="macro", labels=class_values)
            except ValueError:
                auc_roc = np.nan
        average = "binary" if len(class_values) <= 2 else "macro"
        return {
            "local_accuracy": accuracy_score(y_eval, predictions),
            "local_balanced_accuracy": balanced_accuracy_score(y_eval, predictions),
            "local_f1_score": f1_score(y_eval, predictions, average=average, zero_division=0),
            "local_auc_roc": auc_roc,
            "validation_size": len(y_eval),
        }
    except Exception:
        return None


def _spread(values):
    values = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not values:
        return 0.0, 0.0, np.nan
    return float(max(values) - min(values)), float(np.std(values)), float(min(values))


def _client_fairness_records(round_num, clients, params, classes, num_features, epsilons, selected_indices, participation_counts):
    records = []
    local_accuracies = []
    eps_values = []
    for idx, client in enumerate(clients):
        metrics = _evaluate_params_on_client(params, client, classes, num_features)
        epsilon = epsilons[idx] if idx < len(epsilons) else None
        selected = idx in selected_indices
        record = {
            "round": round_num,
            "client_id": client.client_id,
            "selected": selected,
            "participation_count": participation_counts[idx],
            "epsilon": epsilon,
            "validation_size": 0,
            "local_accuracy": np.nan,
            "local_balanced_accuracy": np.nan,
            "local_f1_score": np.nan,
            "local_auc_roc": np.nan,
        }
        if metrics is not None:
            record.update(metrics)
            local_accuracies.append(metrics["local_accuracy"])
        if epsilon is not None and np.isfinite(float(epsilon)):
            eps_values.append(float(epsilon))
        records.append(record)
    acc_gap, acc_std, acc_min = _spread(local_accuracies)
    epsilon_gap, epsilon_std, _ = _spread(eps_values)
    participation_gap, participation_std, _ = _spread(participation_counts)
    summary = {
        "client_accuracy_gap": acc_gap,
        "client_accuracy_std": acc_std,
        "client_min_accuracy": acc_min,
        "epsilon_gap": epsilon_gap,
        "epsilon_std": epsilon_std,
        "participation_gap": participation_gap,
        "participation_std": participation_std,
    }
    return records, summary


def _evaluate_params_global(params, X_eval, y_eval, classes, num_features):
    if X_eval is None or y_eval is None or len(y_eval) == 0:
        return {"accuracy": np.nan, "balanced_accuracy": np.nan, "f1_score": np.nan}
    class_values = np.asarray(classes, dtype=int)
    if "coef_" not in params:
        metrics = _evaluate_softmax_params(params, X_eval, y_eval, classes)
        return {
            "accuracy": metrics.get("accuracy", np.nan),
            "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
            "f1_score": metrics.get("f1_score", np.nan),
        }
    param_classes = 1 if len(class_values) <= 2 else len(class_values)
    model = SGDClassifier(loss="log_loss")
    model.coef_ = np.zeros((param_classes, num_features))
    model.intercept_ = np.zeros(param_classes)
    model.partial_fit(X_eval[:1], y_eval[:1], classes=class_values)
    model.coef_ = np.copy(params["coef_"]).reshape(param_classes, num_features)
    model.intercept_ = np.copy(params["intercept_"]).reshape(param_classes)
    try:
        predictions = model.predict(X_eval)
        average = "binary" if len(class_values) <= 2 else "macro"
        return {
            "accuracy": accuracy_score(y_eval, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_eval, predictions),
            "f1_score": f1_score(y_eval, predictions, average=average, zero_division=0),
        }
    except Exception:
        return {"accuracy": np.nan, "balanced_accuracy": np.nan, "f1_score": np.nan}


def _minmax_normalize(values):
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not finite:
        return {}
    minimum = min(finite)
    maximum = max(finite)
    if abs(maximum - minimum) < 1e-12:
        common = 0.5 if abs(maximum) > 1e-12 else 0.0
        return {idx: common for idx, value in enumerate(values) if value is not None and np.isfinite(float(value))}
    return {
        idx: (float(value) - minimum) / (maximum - minimum)
        for idx, value in enumerate(values)
        if value is not None and np.isfinite(float(value))
    }


def _data_quality_components(client, classes, max_sample_count):
    entropy_score = _compute_data_quality_score(client.y_train)
    sample_score = len(client.y_train) / max(max_sample_count, 1)
    coverage_score = len(np.unique(client.y_train)) / max(len(classes), 1)
    composite_score = float(0.5 * entropy_score + 0.3 * sample_score + 0.2 * coverage_score)
    return {
        "data_entropy_score": float(entropy_score),
        "sample_size_score": float(sample_score),
        "class_coverage_score": float(coverage_score),
        "data_quality_score": composite_score,
    }


def _fairness_penalty_map(fairness_records):
    finite_rows = [
        row for row in fairness_records
        if np.isfinite(_metric_value(row.get("local_accuracy")))
    ]
    if not finite_rows:
        return {}
    mean_accuracy = float(np.mean([_metric_value(row["local_accuracy"]) for row in finite_rows]))
    denominator = max(abs(mean_accuracy), 1e-6)
    return {
        row["client_id"]: min(1.0, max(0.0, (mean_accuracy - _metric_value(row["local_accuracy"])) / denominator))
        for row in finite_rows
    }


def _risk_penalty_from_record(record, quarantine_threshold):
    if record is None:
        return 0.0
    action_penalty = {"normal": 0.0, "warning": 0.25, "downweight": 0.60, "quarantine": 1.0}
    risk_score = _metric_value(record.get("risk_score"))
    if not np.isfinite(risk_score):
        risk_score = 0.0
    risk_component = min(1.0, max(0.0, risk_score / max(quarantine_threshold, 1e-12)))
    return float(max(risk_component, action_penalty.get(record.get("action"), 0.0)))


class ContributionPenaltyEvaluator:
    def __init__(self, args, classes, num_features, config, zkip):
        self.args = args
        self.classes = classes
        self.num_features = num_features
        self.config = config
        self.zkip = zkip
        self.quality_weight = max(0.0, float(args.contribution_quality_weight))
        self.shapley_weight = max(0.0, float(args.contribution_shapley_weight))
        self.risk_weight = max(0.0, float(args.contribution_risk_weight))
        self.fairness_weight = max(0.0, float(args.contribution_fairness_weight))

    def _utility(self, params, X_test, y_test):
        metrics = _evaluate_params_global(params, X_test, y_test, self.classes, self.num_features)
        return _metric_value(metrics.get(self.args.contribution_utility_metric))

    def _aggregate_without_server_noise(self, base_params, updates):
        params, success, _, _, _ = _aggregate_deltas(
            base_params,
            updates,
            self.config["use_zkip"],
            self.zkip,
            privacy_config=None,
            apply_server_dp=False,
        )
        return params if success else base_params

    def evaluate_round(
        self,
        round_num,
        clients,
        selected_indices,
        base_params,
        client_updates,
        adjusted_weights,
        regulatory_records,
        fairness_records,
        round_update_norms,
        X_test,
        y_test,
    ):
        selected_indices = set(selected_indices)
        max_sample_count = max((len(client.y_train) for client in clients), default=1)
        update_by_client = {}
        weighted_updates = []
        for update in client_updates:
            if len(update) == 5:
                delta, proof, client_id, data_size, adjusted_weight = update
            else:
                delta, proof, client_id, data_size = update
                adjusted_weight = adjusted_weights.get(client_id, data_size)
            if delta is None or adjusted_weight <= 0:
                continue
            normalized = (delta, proof, client_id, data_size, adjusted_weight)
            update_by_client[client_id] = normalized
            weighted_updates.append(normalized)

        full_params = self._aggregate_without_server_noise(base_params, weighted_updates)
        full_utility = self._utility(full_params, X_test, y_test)
        if not np.isfinite(full_utility):
            full_utility = 0.0

        shapley_values = [0.0 for _ in clients]
        for idx, client in enumerate(clients):
            if idx not in selected_indices or client.client_id not in update_by_client:
                continue
            without_client = [update for update in weighted_updates if update[2] != client.client_id]
            without_params = self._aggregate_without_server_noise(base_params, without_client)
            without_utility = self._utility(without_params, X_test, y_test)
            without_utility = without_utility if np.isfinite(without_utility) else 0.0
            shapley_values[idx] = float(full_utility - without_utility)

        normalized_shapley = _minmax_normalize(shapley_values)
        fairness_penalties = _fairness_penalty_map(fairness_records)
        regulatory_by_client = {row["client_id"]: row for row in regulatory_records}
        score_values = []
        records = []
        for idx, client in enumerate(clients):
            selected = idx in selected_indices
            quality_components = _data_quality_components(client, self.classes, max_sample_count)
            quality_score = quality_components["data_quality_score"]
            approx_shapley = shapley_values[idx] if selected else 0.0
            shapley_score = normalized_shapley.get(idx, 0.0) if selected else 0.0
            regulatory_record = regulatory_by_client.get(client.client_id)
            risk_penalty = _risk_penalty_from_record(regulatory_record, self.args.reg_quarantine_threshold) if selected else 0.0
            fairness_penalty = fairness_penalties.get(client.client_id, 0.0) if selected else 0.0
            adjusted_weight = 0.0
            action = "not_selected"
            if selected:
                adjusted_weight = adjusted_weights.get(client.client_id, len(client.y_train))
                action = regulatory_record.get("action", "normal") if regulatory_record is not None else "normal"
            final_score = 0.0
            if selected:
                final_score = (
                    self.quality_weight * quality_score
                    + self.shapley_weight * shapley_score
                    - self.risk_weight * risk_penalty
                    - self.fairness_weight * fairness_penalty
                )
                score_values.append(final_score)
            records.append(
                {
                    "round": round_num,
                    "client_id": client.client_id,
                    "selected": selected,
                    **quality_components,
                    "approx_shapley": approx_shapley,
                    "normalized_shapley": shapley_score,
                    "risk_penalty": risk_penalty,
                    "fairness_penalty": fairness_penalty,
                    "final_contribution_score": final_score,
                    "normalized_adjusted_weight": 0.0,
                    "normalized_positive_contribution": 0.0,
                    "contribution_weight_alignment_error": 0.0,
                    "update_norm": round_update_norms[idx] if idx < len(round_update_norms) else None,
                    "adjusted_weight": adjusted_weight,
                    "regulatory_action": action,
                    "utility_metric": self.args.contribution_utility_metric,
                    "full_utility": full_utility,
                }
            )

        selected_records = [record for record in records if record["selected"]]
        total_adjusted_weight = sum(max(0.0, _metric_value(record["adjusted_weight"])) for record in selected_records)
        total_positive_score = sum(max(0.0, _metric_value(record["final_contribution_score"])) for record in selected_records)
        for record in selected_records:
            normalized_weight = (
                max(0.0, _metric_value(record["adjusted_weight"])) / total_adjusted_weight
                if total_adjusted_weight > 0 else 0.0
            )
            normalized_contribution = (
                max(0.0, _metric_value(record["final_contribution_score"])) / total_positive_score
                if total_positive_score > 0 else 0.0
            )
            record["normalized_adjusted_weight"] = normalized_weight
            record["normalized_positive_contribution"] = normalized_contribution
            record["contribution_weight_alignment_error"] = abs(normalized_contribution - normalized_weight)

        score_gap, score_std, _ = _spread(score_values)
        shapley_gap, shapley_std, _ = _spread([record["approx_shapley"] for record in records if record["selected"]])
        alignment_errors = [
            record["contribution_weight_alignment_error"]
            for record in records
            if record["selected"] and np.isfinite(_metric_value(record["contribution_weight_alignment_error"]))
        ]
        summary = {
            "avg_contribution_score": float(np.mean(score_values)) if score_values else 0.0,
            "contribution_score_gap": score_gap,
            "contribution_score_std": score_std,
            "avg_approx_shapley": float(np.mean([record["approx_shapley"] for record in records if record["selected"]])) if score_values else 0.0,
            "approx_shapley_gap": shapley_gap,
            "approx_shapley_std": shapley_std,
            "avg_risk_penalty": float(np.mean([record["risk_penalty"] for record in records if record["selected"]])) if score_values else 0.0,
            "avg_fairness_penalty": float(np.mean([record["fairness_penalty"] for record in records if record["selected"]])) if score_values else 0.0,
            "avg_weight_alignment_error": float(np.mean(alignment_errors)) if alignment_errors else 0.0,
            "max_weight_alignment_error": float(max(alignment_errors)) if alignment_errors else 0.0,
        }
        return records, summary


def _parse_csv_list(value, allowed, name):
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in items if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported {name}: {invalid}. Supported: {allowed}")
    return items


def _parse_client_indices(value, num_clients):
    indices = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= num_clients:
            raise ValueError(f"Invalid polluted client index {idx}. Expected 0 <= idx < {num_clients}.")
        indices.append(idx)
    return sorted(set(indices))


def _pollution_active(args, round_num):
    if args.experiment_suite != "pollution" or not args.enable_pollution_injection:
        return False
    end_round = args.pollution_end_round if args.pollution_end_round > 0 else args.num_rounds
    return args.pollution_start_round <= round_num <= end_round


def _apply_client_pollution(client, classes, args, rng):
    sample_count = len(client.y_train)
    if sample_count == 0:
        return None, None, None, 0
    polluted_count = max(1, int(np.ceil(sample_count * args.pollution_rate)))
    polluted_count = min(polluted_count, sample_count)
    sample_indices = rng.choice(sample_count, size=polluted_count, replace=False)
    original_X = np.copy(client.X_train)
    original_y = np.copy(client.y_train)

    if args.pollution_type == "label_flip":
        class_values = np.asarray(classes, dtype=int)
        class_to_pos = {label: pos for pos, label in enumerate(class_values)}
        flipped = np.copy(client.y_train[sample_indices])
        for i, label in enumerate(flipped):
            pos = class_to_pos.get(int(label), 0)
            flipped[i] = class_values[(pos + 1) % len(class_values)]
        client.y_train[sample_indices] = flipped
    elif args.pollution_type == "feature_noise":
        noise = rng.normal(0, args.pollution_feature_noise_std, size=client.X_train[sample_indices].shape)
        client.X_train[sample_indices] = client.X_train[sample_indices] + noise
    else:
        raise ValueError(f"Unsupported pollution type: {args.pollution_type}")
    return original_X, original_y, sample_indices, polluted_count


def _restore_client_data(client, original_X, original_y):
    if original_X is not None:
        client.X_train = original_X
    if original_y is not None:
        client.y_train = original_y


def _select_participants(policy, available_indices, clients, capabilities, contribution_scores, participation_rate, rng):
    if not available_indices:
        return []
    if policy == "all":
        return list(available_indices)
    target_count = max(1, int(np.ceil(len(available_indices) * participation_rate)))
    target_count = min(target_count, len(available_indices))
    if policy == "random":
        return sorted(rng.choice(available_indices, size=target_count, replace=False).tolist())
    if policy != "apdp_score":
        raise ValueError(f"Unsupported participation policy: {policy}")

    max_size = max(len(clients[i].y_train) for i in available_indices) or 1
    max_contribution = max(abs(contribution_scores.get(i, 0.0)) for i in available_indices) or 1.0
    scores = []
    for idx in available_indices:
        data_score = len(clients[idx].y_train) / max_size
        quality_score = _compute_data_quality_score(clients[idx].y_train)
        compute_score = capabilities.get(idx, 1.0)
        contribution_score = abs(contribution_scores.get(idx, 0.0)) / max_contribution
        score = 0.35 * data_score + 0.25 * quality_score + 0.20 * compute_score + 0.20 * contribution_score
        scores.append((score, idx))
    return [idx for _, idx in sorted(scores, reverse=True)[:target_count]]


def _client_idx_from_id(client_id):
    try:
        return int(str(client_id).split("_")[-1])
    except (TypeError, ValueError):
        return None


def _mean_or_nan(values):
    clean = []
    for value in values:
        if value is None or value == "":
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            clean.append(numeric)
    return float(np.mean(clean)) if clean else np.nan


def _build_resource_privacy_diagnostics(result, method_name, client_ids):
    profiles = {int(row["client_idx"]): row for row in result.get("resource_profiles", [])}
    trace_rows = result.get("resource_trace_records", [])
    decisions = result.get("orchestration_decisions", [])
    privacy_rows = result.get("privacy_accounting_records", [])
    partial_rows = result.get("partial_update_records", [])
    if not profiles and not trace_rows and not decisions and not privacy_rows:
        return [], []

    statuses_by_client = {idx: {} for idx in range(len(client_ids))}
    for row in trace_rows:
        idx = int(row.get("client_idx", -1))
        if idx < 0:
            continue
        status = row.get("status", "")
        statuses_by_client.setdefault(idx, {})[status] = statuses_by_client.setdefault(idx, {}).get(status, 0) + 1

    decisions_by_client = {idx: [] for idx in range(len(client_ids))}
    for row in decisions:
        idx = _client_idx_from_id(row.get("client_id"))
        if idx is not None:
            decisions_by_client.setdefault(idx, []).append(row)

    privacy_by_client = {idx: [] for idx in range(len(client_ids))}
    for row in privacy_rows:
        idx = _client_idx_from_id(row.get("client_id"))
        if idx is not None:
            privacy_by_client.setdefault(idx, []).append(row)

    partial_by_client = {idx: [] for idx in range(len(client_ids))}
    for row in partial_rows:
        idx = _client_idx_from_id(row.get("client_id"))
        if idx is not None:
            partial_by_client.setdefault(idx, []).append(row)

    selected_counts = {idx: statuses.get("selected", 0) for idx, statuses in statuses_by_client.items()}
    max_selected = max(selected_counts.values()) if selected_counts else 0
    diagnostics = []
    for idx, client_id in enumerate(client_ids):
        statuses = statuses_by_client.get(idx, {})
        selected = statuses.get("selected", 0)
        not_selected = statuses.get("not_selected", 0)
        deadline_fail = statuses.get("deadline_infeasible", 0)
        privacy_fail = statuses.get("privacy_budget_infeasible", 0)
        unavailable = statuses.get("unavailable", 0)
        quarantine = statuses.get("quarantine", 0)
        failure_plan = statuses.get("failure_plan", 0)
        deadline_denominator = selected + not_selected + deadline_fail
        eligible_denominator = selected + not_selected + deadline_fail + privacy_fail
        decision_rows = decisions_by_client.get(idx, [])
        privacy_client_rows = privacy_by_client.get(idx, [])
        partial_client_rows = partial_by_client.get(idx, [])
        latest_privacy = privacy_client_rows[-1] if privacy_client_rows else {}
        target_epsilon = float(latest_privacy.get("target_epsilon", np.nan)) if latest_privacy else np.nan
        final_epsilon = float(latest_privacy.get("cumulative_epsilon", 0.0)) if latest_privacy else 0.0
        diagnostics.append({
            "method": method_name,
            "client_id": client_id,
            "client_idx": idx,
            "tier": profiles.get(idx, {}).get("tier", ""),
            "trace_events": sum(statuses.values()),
            "selected_count": selected,
            "not_selected_count": not_selected,
            "deadline_infeasible_count": deadline_fail,
            "privacy_budget_infeasible_count": privacy_fail,
            "unavailable_count": unavailable,
            "quarantine_count": quarantine,
            "failure_plan_count": failure_plan,
            "selection_rate": selected / max(1, sum(statuses.values())),
            "deadline_feasible_rate": (selected + not_selected) / max(1, deadline_denominator),
            "historical_success_rate": selected / max(1, eligible_denominator),
            "tier_participation_debt": max_selected - selected,
            "target_epsilon": target_epsilon,
            "final_epsilon": final_epsilon,
            "remaining_epsilon": float(latest_privacy.get("remaining_epsilon", np.nan)) if latest_privacy else np.nan,
            "epsilon_utilization": final_epsilon / target_epsilon if target_epsilon and np.isfinite(target_epsilon) else np.nan,
            "spent_events": sum(row.get("status") == "spent" for row in privacy_client_rows),
            "budget_exhausted_events": sum(row.get("status") == "budget_exhausted" for row in privacy_client_rows),
            "avg_noise_multiplier": _mean_or_nan([row.get("noise_multiplier") for row in decision_rows]),
            "avg_incremental_epsilon": _mean_or_nan([row.get("incremental_epsilon") for row in privacy_client_rows]),
            "avg_privacy_budget_target": _mean_or_nan([row.get("privacy_budget_target") for row in decision_rows]),
            "avg_expected_future_opportunities": _mean_or_nan([row.get("expected_future_opportunities") for row in decision_rows]),
            "avg_budget_utilization_at_selection": _mean_or_nan([row.get("budget_utilization") for row in decision_rows]),
            "avg_privacy_boost": _mean_or_nan([row.get("privacy_boost") for row in decision_rows]),
            "avg_opportunity_compensation": _mean_or_nan([row.get("opportunity_compensation") for row in decision_rows]),
            "avg_upload_ratio": _mean_or_nan([row.get("upload_ratio") for row in decision_rows]),
            "avg_deadline_slack_ratio": _mean_or_nan([row.get("deadline_slack_ratio") for row in decision_rows]),
            "avg_residual_pressure": _mean_or_nan([row.get("residual_pressure") for row in decision_rows]),
            "compressed_selection_count": sum(
                row.get("upload_selection_reason") in {"compressed_to_restore_deadline_slack", "compressed_best_effort_deadline"}
                for row in decision_rows
            ),
            "safe_full_upload_count": sum(row.get("upload_selection_reason") == "full_upload_with_safe_slack" for row in decision_rows),
            "residual_feedback_full_upload_count": sum(row.get("upload_selection_reason") == "residual_feedback_full_upload" for row in decision_rows),
            "uploaded_bytes": sum(int(row.get("uploaded_bytes", 0)) for row in partial_client_rows),
            "avg_uploaded_parameter_fraction": _mean_or_nan([row.get("uploaded_parameter_fraction") for row in partial_client_rows]),
            "avg_residual_l2_before": _mean_or_nan([row.get("residual_l2_before") for row in partial_client_rows]),
            "mean_residual_l2": _mean_or_nan([row.get("residual_l2") for row in partial_client_rows]),
            "avg_residual_l2_after": _mean_or_nan([row.get("residual_l2_after") for row in partial_client_rows]),
        })

    tier_summary = []
    for tier in sorted({row["tier"] for row in diagnostics if row.get("tier")}):
        rows = [row for row in diagnostics if row.get("tier") == tier]
        tier_summary.append({
            "method": method_name,
            "tier": tier,
            "client_count": len(rows),
            "selected_count": sum(row["selected_count"] for row in rows),
            "deadline_infeasible_count": sum(row["deadline_infeasible_count"] for row in rows),
            "privacy_budget_infeasible_count": sum(row["privacy_budget_infeasible_count"] for row in rows),
            "failure_plan_count": sum(row["failure_plan_count"] for row in rows),
            "avg_selection_rate": _mean_or_nan([row["selection_rate"] for row in rows]),
            "avg_deadline_feasible_rate": _mean_or_nan([row["deadline_feasible_rate"] for row in rows]),
            "avg_historical_success_rate": _mean_or_nan([row["historical_success_rate"] for row in rows]),
            "avg_tier_participation_debt": _mean_or_nan([row["tier_participation_debt"] for row in rows]),
            "avg_final_epsilon": _mean_or_nan([row["final_epsilon"] for row in rows]),
            "avg_epsilon_utilization": _mean_or_nan([row["epsilon_utilization"] for row in rows]),
            "avg_noise_multiplier": _mean_or_nan([row["avg_noise_multiplier"] for row in rows]),
            "avg_privacy_budget_target": _mean_or_nan([row["avg_privacy_budget_target"] for row in rows]),
            "avg_opportunity_compensation": _mean_or_nan([row["avg_opportunity_compensation"] for row in rows]),
            "avg_upload_ratio": _mean_or_nan([row["avg_upload_ratio"] for row in rows]),
            "avg_deadline_slack_ratio": _mean_or_nan([row["avg_deadline_slack_ratio"] for row in rows]),
            "avg_residual_pressure": _mean_or_nan([row["avg_residual_pressure"] for row in rows]),
            "compressed_selection_count": sum(row["compressed_selection_count"] for row in rows),
            "safe_full_upload_count": sum(row["safe_full_upload_count"] for row in rows),
            "residual_feedback_full_upload_count": sum(row["residual_feedback_full_upload_count"] for row in rows),
            "avg_uploaded_parameter_fraction": _mean_or_nan([row["avg_uploaded_parameter_fraction"] for row in rows]),
            "avg_residual_l2_before": _mean_or_nan([row["avg_residual_l2_before"] for row in rows]),
            "avg_residual_l2_after": _mean_or_nan([row["avg_residual_l2_after"] for row in rows]),
            "uploaded_bytes": sum(row["uploaded_bytes"] for row in rows),
        })
    return diagnostics, tier_summary


def _save_method_artifacts(output_dir, method_name, result, client_ids):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for idx, round_num in enumerate(result["rounds"]):
        rows.append(
            {
                "method": method_name,
                "round": round_num,
                "accuracy": result["accuracies"][idx],
                "balanced_accuracy": result["balanced_accuracies"][idx],
                "f1_score": result["f1_scores"][idx],
                "precision": result["precisions"][idx],
                "recall": result["recalls"][idx],
                "auc_roc": result["aucs"][idx],
                "round_duration": result["round_durations"][idx],
                "agg_client_count": result["agg_client_counts"][idx],
                "selected_client_count": result.get("selected_client_counts", [np.nan] * len(result["rounds"]))[idx],
                "participation_policy": result.get("participation_policy", ""),
                "dp_noise_scale": result["dp_noise_scales"][idx],
                "zkip_failures": result["zkip_failures"][idx],
                "ebcd_alert": result["ebcd_alerts"][idx],
                "tcm_state_count": result["tcm_counts"][idx],
            }
        )

    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["method", "round"])
        writer.writeheader()
        writer.writerows(rows)

    serializable_result = {key: value for key, value in result.items() if key != "tcm"}
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(serializable_result), f, indent=2)

    np.save(os.path.join(output_dir, "per_client_update_norms.npy"), np.array(result["per_client_update_norms"], dtype=object))
    np.save(os.path.join(output_dir, "per_client_ebcd_stats.npy"), np.array(result["per_client_ebcd_stats"], dtype=object))
    np.save(os.path.join(output_dir, "per_client_zkip_status.npy"), np.array(result["per_client_zkip_status"], dtype=object))
    np.save(os.path.join(output_dir, "per_client_epsilon.npy"), np.array(result["per_client_epsilon"], dtype=object))
    np.save(os.path.join(output_dir, "per_client_noise_multiplier.npy"), np.array(result.get("per_client_noise_multiplier", []), dtype=object))
    privacy_rows = result.get("privacy_accounting_records", [])
    if privacy_rows:
        _write_csv(os.path.join(output_dir, "privacy_accounting.csv"), privacy_rows)
        summaries = []
        for client_id in sorted({row["client_id"] for row in privacy_rows}):
            rows_for_client = [row for row in privacy_rows if row["client_id"] == client_id]
            latest = rows_for_client[-1]
            summaries.append({
                "method": method_name,
                "client_id": client_id,
                "target_epsilon": latest["target_epsilon"],
                "final_epsilon": latest["cumulative_epsilon"],
                "remaining_epsilon": latest["remaining_epsilon"],
                "spent_events": sum(row["status"] == "spent" for row in rows_for_client),
                "budget_exhausted_events": sum(row["status"] == "budget_exhausted" for row in rows_for_client),
            })
        _write_csv(os.path.join(output_dir, "privacy_accounting_summary.csv"), summaries)
        plt.figure(figsize=charts.FIGSIZE_WIDE)
        epsilon_history = np.asarray(result["per_client_epsilon"], dtype=object)
        for client_idx, client_id in enumerate(client_ids):
            values = [row[client_idx] if client_idx < len(row) else np.nan for row in epsilon_history]
            plt.plot(result["rounds"], values, label=client_id, linewidth=1)
        plt.axhline(summaries[0]["target_epsilon"], color="black", linestyle="--", label="target epsilon")
        plt.xlabel("Round")
        plt.ylabel("Cumulative epsilon")
        plt.title("Per-client RDP privacy spending")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "per_client_cumulative_epsilon.png"))
        plt.close()
    if result.get("tcm") is not None:
        export_tcm_checkpoints(output_dir, result["tcm"])
    if result.get("resource_profiles"):
        _write_csv(os.path.join(output_dir, "resource_profiles.csv"), result["resource_profiles"])
    if result.get("resource_trace_records"):
        _write_csv(os.path.join(output_dir, "resource_trace.csv"), result["resource_trace_records"])
    if result.get("orchestration_decisions"):
        _write_csv(os.path.join(output_dir, "orchestration_decisions.csv"), result["orchestration_decisions"])
    diagnostics, tier_summary = _build_resource_privacy_diagnostics(result, method_name, client_ids)
    if diagnostics:
        _write_csv(os.path.join(output_dir, "resource_privacy_diagnostics.csv"), diagnostics)
    if tier_summary:
        _write_csv(os.path.join(output_dir, "tier_privacy_summary.csv"), tier_summary)
        tiers = [row["tier"] for row in tier_summary]
        x = np.arange(len(tiers))
        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        plt.bar(x, [row["avg_epsilon_utilization"] for row in tier_summary])
        plt.xticks(x, tiers)
        plt.ylim(0.0, 1.05)
        plt.ylabel("Average epsilon utilization")
        plt.title("ARPA Privacy-budget Utilization by Resource Tier")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "tier_epsilon_utilization.png"))
        plt.close()

        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        plt.bar(x, [row["avg_historical_success_rate"] for row in tier_summary])
        plt.xticks(x, tiers)
        plt.ylim(0.0, 1.05)
        plt.ylabel("Average selected / eligible rate")
        plt.title("ARPA Effective Participation by Resource Tier")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "tier_effective_participation.png"))
        plt.close()

        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        plt.bar(x, [row["avg_upload_ratio"] for row in tier_summary])
        plt.xticks(x, tiers)
        plt.ylim(0.0, 1.05)
        plt.ylabel("Average upload ratio")
        plt.title("ARPA Partial-update Ratio by Resource Tier")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "tier_upload_ratio.png"))
        plt.close()
    if result.get("partial_update_records"):
        _write_csv(os.path.join(output_dir, "partial_update_metrics.csv"), result["partial_update_records"])
        partial_rows = result["partial_update_records"]
        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        by_round = {}
        for row in partial_rows:
            by_round[row["round"]] = by_round.get(row["round"], 0) + row["uploaded_bytes"]
        plt.plot(sorted(by_round), [by_round[round_num] for round_num in sorted(by_round)], marker="o")
        plt.xlabel("Round")
        plt.ylabel("Uploaded parameter bytes")
        plt.title("ARPA Partial-update Communication Load")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "partial_update_upload.png"))
        plt.close()
    if result.get("resource_trace_records"):
        selected_rows = [row for row in result["resource_trace_records"] if row.get("status") == "selected"]
        if selected_rows:
            completion = {}
            for row in selected_rows:
                completion.setdefault(row["round"], []).append(row["predicted_total_seconds"] <= result["resource_deadline_seconds"])
            plt.figure(figsize=charts.FIGSIZE_DEFAULT)
            plt.plot(sorted(completion), [np.mean(completion[round_num]) for round_num in sorted(completion)], marker="o")
            plt.ylim(-0.05, 1.05)
            plt.xlabel("Round")
            plt.ylabel("Predicted deadline completion rate")
            plt.title("ARPA Resource Deadline Completion")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            charts.save_figure(os.path.join(output_dir, "resource_deadline_completion.png"))
            plt.close()

    charts.plot_global_metrics(result["rounds"], result["accuracies"], result["f1_scores"], result["aucs"])
    charts.save_figure(os.path.join(output_dir, "global_metrics.png"))
    plt.close()
    charts.plot_dp_noise_scale(result["rounds"], result["dp_noise_scales"])
    charts.save_figure(os.path.join(output_dir, "dp_noise_scale.png"))
    plt.close()
    charts.plot_agg_client_counts(result["rounds"], result["agg_client_counts"])
    charts.save_figure(os.path.join(output_dir, "agg_client_counts.png"))
    plt.close()
    charts.plot_zkip_failures(result["rounds"], result["zkip_failures"])
    charts.save_figure(os.path.join(output_dir, "zkip_failures.png"))
    plt.close()
    charts.plot_delta_norm(result["rounds"], result["delta_norms"])
    charts.save_figure(os.path.join(output_dir, "delta_norm.png"))
    plt.close()
    charts.plot_ebcd_alerts(result["rounds"], result["ebcd_alerts"])
    charts.save_figure(os.path.join(output_dir, "ebcd_alerts.png"))
    plt.close()
    charts.plot_tcm_state_count(result["rounds"], result["tcm_counts"])
    charts.save_figure(os.path.join(output_dir, "tcm_state_count.png"))
    plt.close()
    charts.plot_per_client_update_norms(result["rounds"], result["per_client_update_norms"], client_ids)
    charts.save_figure(os.path.join(output_dir, "per_client_update_norms.png"))
    plt.close()

    if result.get("regulatory_enabled"):
        regulatory_rows = result.get("regulatory_records", [])
        fieldnames = [
            "round",
            "client_id",
            "risk_score",
            "action",
            "original_weight",
            "adjusted_weight",
            "zkip_status",
            "update_norm",
            "ebcd_variance",
            "ebcd_kurtosis",
            "ebcd_skewness",
        ]
        with open(os.path.join(output_dir, "regulatory_intervention_summary.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(regulatory_rows)

        rounds = result["rounds"]
        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        plt.plot(rounds, result["regulatory_warning_counts"], marker="o", label="warning")
        plt.plot(rounds, result["regulatory_downweight_counts"], marker="o", label="downweight")
        plt.plot(rounds, result["regulatory_quarantine_counts"], marker="o", label="quarantine")
        plt.xlabel("Round")
        plt.ylabel("Client Count")
        plt.title("Regulatory Intervention Actions")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "regulatory_actions.png"))
        plt.close()

        plt.figure(figsize=charts.FIGSIZE_WIDE)
        for client_id in client_ids:
            rows = [row for row in regulatory_rows if row["client_id"] == client_id]
            if rows:
                plt.plot([row["round"] for row in rows], [row["risk_score"] for row in rows], marker="o", label=client_id)
        plt.xlabel("Round")
        plt.ylabel("Risk Score")
        plt.title("Regulatory Risk by Client")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "regulatory_risk_by_client.png"))
        plt.close()

    if result.get("pollution_enabled"):
        pollution_rows = result.get("pollution_records", [])
        fieldnames = ["round", "client_id", "pollution_type", "polluted_sample_count", "pollution_rate"]
        with open(os.path.join(output_dir, "pollution_injection_summary.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pollution_rows)

    if result.get("fairness_enabled"):
        fairness_rows = result.get("fairness_records", [])
        fieldnames = [
            "round",
            "client_id",
            "selected",
            "participation_count",
            "epsilon",
            "validation_size",
            "local_accuracy",
            "local_balanced_accuracy",
            "local_f1_score",
            "local_auc_roc",
        ]
        with open(os.path.join(output_dir, "client_fairness_summary.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fairness_rows)

        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        plt.plot(result["rounds"], result["client_accuracy_gaps"], marker="o", label="accuracy gap")
        plt.plot(result["rounds"], result["client_accuracy_stds"], marker="o", label="accuracy std")
        plt.xlabel("Round")
        plt.ylabel("Client Accuracy Disparity")
        plt.title("Client Performance Fairness")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "client_performance_fairness.png"))
        plt.close()

        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        plt.plot(result["rounds"], result["epsilon_gaps"], marker="o", label="epsilon gap")
        plt.plot(result["rounds"], result["epsilon_stds"], marker="o", label="epsilon std")
        plt.xlabel("Round")
        plt.ylabel("Privacy Budget Disparity")
        plt.title("Privacy Budget Fairness")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "privacy_budget_fairness.png"))
        plt.close()

        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        plt.plot(result["rounds"], result["participation_gaps"], marker="o", label="participation gap")
        plt.plot(result["rounds"], result["participation_stds"], marker="o", label="participation std")
        plt.xlabel("Round")
        plt.ylabel("Participation Disparity")
        plt.title("Client Participation Fairness")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "participation_fairness.png"))
        plt.close()

    if result.get("contribution_enabled"):
        contribution_rows = result.get("contribution_records", [])
        _write_csv(os.path.join(output_dir, "contribution_penalty_summary.csv"), contribution_rows)

        plt.figure(figsize=charts.FIGSIZE_WIDE)
        for client_id in client_ids:
            rows = [row for row in contribution_rows if row["client_id"] == client_id]
            if rows:
                plt.plot(
                    [row["round"] for row in rows],
                    [row["final_contribution_score"] for row in rows],
                    marker="o",
                    label=client_id,
                )
        plt.xlabel("Round")
        plt.ylabel("Contribution Score")
        plt.title("Contribution Score by Client")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "contribution_score_by_client.png"))
        plt.close()

        plt.figure(figsize=charts.FIGSIZE_WIDE)
        for client_id in client_ids:
            rows = [row for row in contribution_rows if row["client_id"] == client_id and row["selected"]]
            if rows:
                plt.plot(
                    [row["round"] for row in rows],
                    [row["approx_shapley"] for row in rows],
                    marker="o",
                    label=client_id,
                )
        plt.xlabel("Round")
        plt.ylabel("Approximate Shapley")
        plt.title("Leave-One-Out Approximate Shapley by Client")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "approx_shapley_by_client.png"))
        plt.close()

        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        plt.plot(result["rounds"], result.get("avg_risk_penalties", []), marker="o", label="risk penalty")
        plt.plot(result["rounds"], result.get("avg_fairness_penalties", []), marker="o", label="fairness penalty")
        plt.xlabel("Round")
        plt.ylabel("Average Penalty")
        plt.title("Penalty Components")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "penalty_components.png"))
        plt.close()

        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        plt.plot(result["rounds"], result.get("avg_weight_alignment_errors", []), marker="o", label="average error")
        plt.plot(result["rounds"], result.get("max_weight_alignment_errors", []), marker="o", label="max error")
        plt.xlabel("Round")
        plt.ylabel("Alignment Error")
        plt.title("Contribution-Weight Alignment")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "contribution_weight_alignment.png"))
        plt.close()
    write_artifact_manifest(output_dir)


def _pollution_detection_metrics(result):
    polluted_pairs = {(row["round"], row["client_id"]) for row in result.get("pollution_records", [])}
    regulatory_rows = result.get("regulatory_records", [])
    actionable = {"warning", "downweight", "quarantine"}
    detected_pairs = {
        (row["round"], row["client_id"])
        for row in regulatory_rows
        if (row["round"], row["client_id"]) in polluted_pairs and row["action"] in actionable
    }
    false_positive_pairs = {
        (row["round"], row["client_id"])
        for row in regulatory_rows
        if (row["round"], row["client_id"]) not in polluted_pairs and row["action"] in actionable
    }
    total_polluted = len(polluted_pairs)
    return {
        "total_polluted_client_rounds": total_polluted,
        "detected_polluted_client_rounds": len(detected_pairs),
        "pollution_detection_rate": len(detected_pairs) / total_polluted if total_polluted else 0.0,
        "regulatory_false_positive_actions": len(false_positive_pairs),
    }


def _run_single_method(
    method_name,
    args,
    train_val_data,
    X_test,
    y_test,
    classes,
    failure_plan,
    output_dir,
    privacy_config,
    participation_policy="all",
    participation_rate=1.0,
    label_suffix="",
    synthetic_metadata=None,
    capability_overrides=None,
    config_overrides=None,
    ablation_scenario=None,
):
    config = copy.deepcopy(METHOD_CONFIGS[method_name])
    if config_overrides:
        config.update(config_overrides)
    num_features = X_test.shape[1]
    clients, backend, backend_device = _init_backend_clients(train_val_data, num_features, classes, args, privacy_config)
    client_ids = [client.client_id for client in clients]
    capabilities = capability_overrides if capability_overrides is not None else _assign_capabilities(len(clients))
    for i, client in enumerate(clients):
        client.compute_capability = capabilities.get(i, 1.0)

    server = FLServer(f"{method_name}_server", client_ids, num_features, classes=classes)
    if backend == "torch":
        server.global_model_parameters = (
            {key: np.copy(value) for key, value in clients[0].model_parameters().items()}
            if clients
            else {
                "coef_": np.zeros((len(classes), num_features), dtype=np.float32),
                "intercept_": np.zeros(len(classes), dtype=np.float32),
            }
        )
    if config["use_ebcd"]:
        initial_params = [client.model_parameters() for client in clients if client.X_train.shape[0] > 0]
        if initial_params:
            server.ebcd.establish_baseline(initial_params)

    active_indices = list(range(len(clients)))
    current_allocations = _allocate_budget(clients, active_indices, capabilities, config["dynamic_privacy"], privacy_config)
    client_privacy_state = _make_client_privacy_state(clients, args, privacy_config, config) if config["dp_scope"] == "client" else {}
    resource_simulation_enabled = getattr(args, "heterogeneity_profile", "legacy") == "regulated_generic"
    arpa_enabled = resource_simulation_enabled and config.get("resource_orchestrator", False)
    resource_orchestrator = None
    resource_profiles = []
    if resource_simulation_enabled:
        if (args.round_deadline_seconds <= 0 or args.reference_batch_seconds <= 0
                or args.parameter_blocks <= 0 or args.arpa_min_initial_privacy_spend <= 0
                or args.arpa_privacy_boost_gain < 0 or args.arpa_max_privacy_boost < 1
                or args.arpa_opportunity_compensation_weight < 0
                or not 0 < args.arpa_compression_slack_target <= 1
                or not 0 <= args.arpa_residual_full_upload_threshold <= 1):
            raise ValueError(
                "ARPA deadline, reference duration, parameter blocks, initial privacy spend, "
                "privacy boost, and opportunity compensation parameters must be valid"
            )
        profiles = build_resource_profiles(len(clients), args.seed)
        resource_orchestrator = ResourcePrivacyOrchestrator(
            profiles=profiles,
            seed=args.seed,
            deadline_seconds=args.round_deadline_seconds,
            reference_batch_seconds=args.reference_batch_seconds,
            upload_ratios=(1.0,) if not arpa_enabled or config.get("force_full_upload", False) else _parse_upload_ratios(args.upload_ratios),
            block_count=args.parameter_blocks,
            enforce_tier_coverage=config.get("resource_fairness", True),
            minimum_initial_privacy_increment=args.arpa_min_initial_privacy_spend,
            enable_opportunity_privacy=config.get("opportunity_privacy", True),
            enable_budget_utilization_boost=config.get("budget_utilization_boost", True),
            enable_low_resource_compensation=config.get("low_resource_compensation", True),
            privacy_boost_gain=args.arpa_privacy_boost_gain,
            max_privacy_boost=args.arpa_max_privacy_boost,
            opportunity_compensation_weight=args.arpa_opportunity_compensation_weight,
            compression_slack_target=args.arpa_compression_slack_target,
            residual_full_upload_threshold=args.arpa_residual_full_upload_threshold,
        )
        resource_profiles = [profile.__dict__.copy() for profile in profiles.values()]
    residuals = {
        idx: {key: np.zeros_like(value) for key, value in server.global_model_parameters.items()}
        for idx in range(len(clients))
    }
    previous_risk_actions = {}
    contribution_history = []
    server_optimizer_state = {} if config.get("aggregation") == "fedadam" else None

    result = {
        "method": method_name,
        "label": f"{config['label']}{label_suffix}" if backend == "sklearn" else f"{config['label']}{label_suffix} (torch)",
        "backend": backend,
        "device": str(backend_device) if backend_device is not None else "",
        "reference": config.get("reference", ""),
        "participation_policy": participation_policy,
        "ablation_scenario": ablation_scenario or "",
        "regulatory_enabled": (
            args.experiment_suite in {"contribution", "audit_trace"}
            or (args.enable_regulatory_intervention and args.experiment_suite in {"baselines", "pollution", "synthetic_fairness", "contribution", "audit_trace", "ablation"})
        ),
        "pollution_enabled": args.experiment_suite == "pollution" and args.enable_pollution_injection,
        "fairness_enabled": args.experiment_suite in {"fairness", "synthetic_fairness", "contribution", "audit_trace"} or args.enable_fairness_evaluation,
        "contribution_enabled": args.experiment_suite in {"contribution", "audit_trace"} or args.enable_contribution_evaluation,
        "synthetic_sensitive_metadata": synthetic_metadata or [],
        "rounds": [],
        "accuracies": [],
        "balanced_accuracies": [],
        "f1_scores": [],
        "precisions": [],
        "recalls": [],
        "aucs": [],
        "round_durations": [],
        "agg_client_counts": [],
        "selected_client_counts": [],
        "dp_noise_scales": [],
        "zkip_failures": [],
        "delta_norms": [],
        "ebcd_alerts": [],
        "tcm_counts": [],
        "per_client_update_norms": [],
        "per_client_ebcd_stats": [],
        "per_client_zkip_status": [],
        "per_client_epsilon": [],
        "per_client_noise_multiplier": [],
        "privacy_accounting_records": [],
        "tcm": server.tcm,
        "regulatory_records": [],
        "regulatory_warning_counts": [],
        "regulatory_downweight_counts": [],
        "regulatory_quarantine_counts": [],
        "regulatory_avg_risks": [],
        "pollution_records": [],
        "fairness_records": [],
        "client_accuracy_gaps": [],
        "client_accuracy_stds": [],
        "client_min_accuracies": [],
        "epsilon_gaps": [],
        "epsilon_stds": [],
        "participation_gaps": [],
        "participation_stds": [],
        "contribution_records": [],
        "avg_contribution_scores": [],
        "contribution_score_gaps": [],
        "contribution_score_stds": [],
        "avg_approx_shapley": [],
        "approx_shapley_gaps": [],
        "approx_shapley_stds": [],
        "avg_risk_penalties": [],
        "avg_fairness_penalties": [],
        "avg_weight_alignment_errors": [],
        "max_weight_alignment_errors": [],
        "final_group_accuracy_gap": 0.0,
        "final_worst_group_accuracy": np.nan,
        "final_group_f1_gap": 0.0,
        "final_group_epsilon_gap": 0.0,
        "final_group_participation_gap": 0.0,
        "final_group_quarantine_gap": 0.0,
        "resource_profiles": resource_profiles,
        "resource_trace_records": [],
        "orchestration_decisions": [],
        "partial_update_records": [],
        "resource_deadline_seconds": args.round_deadline_seconds if resource_simulation_enabled else None,
    }

    backend_description = f", backend={backend}" + (f", device={backend_device}" if backend_device is not None else "")
    print(f"\n=== Running {config['label']} ({method_name}{backend_description}) ===")
    rng = np.random.default_rng(args.seed + 5000)
    contribution_scores = {idx: 0.0 for idx in range(len(clients))}
    regulatory_controller = None
    if result["regulatory_enabled"]:
        regulatory_controller = RegulatoryInterventionController(
            warning_threshold=args.reg_warning_threshold,
            quarantine_threshold=args.reg_quarantine_threshold,
            penalty_weight=args.reg_penalty_weight,
        )
    contribution_evaluator = None
    if result["contribution_enabled"]:
        contribution_evaluator = ContributionPenaltyEvaluator(args, classes, num_features, config, server.zkip)
    polluted_indices = _parse_client_indices(args.polluted_clients, len(clients)) if result["pollution_enabled"] else []
    pollution_rng = np.random.default_rng(args.seed + 9000)
    participation_counts = [0 for _ in clients]
    for round_num in range(1, args.num_rounds + 1):
        start_time = time.time()
        global_params = {k: np.copy(v) for k, v in server.global_model_parameters.items()}
        client_updates = []
        round_update_norms = []
        round_ebcd_stats = []
        round_zkip_status = []
        round_epsilons = []
        round_noise_multipliers = []
        round_noise_scales = []
        round_delta_norm = 0.0
        masks_by_client = {}
        available_indices = [idx for idx in range(len(clients)) if not failure_plan[round_num - 1][idx]]
        planned_actions = {}
        if arpa_enabled:
            target_count = max(1, int(np.ceil(len(available_indices) * participation_rate))) if available_indices else 0
            quality_scores = {idx: _compute_data_quality_score(client.y_train) for idx, client in enumerate(clients)}
            model_norm = float(np.sqrt(sum(np.linalg.norm(value) ** 2 for value in global_params.values())))
            residual_pressures = {
                idx: (
                    float(np.sqrt(sum(np.linalg.norm(value) ** 2 for value in residuals[idx].values())))
                    / max(1e-12, model_norm + float(np.sqrt(sum(np.linalg.norm(value) ** 2 for value in residuals[idx].values()))))
                )
                for idx in range(len(clients))
            }
            planned_actions, trace_rows = resource_orchestrator.plan(
                round_num=round_num,
                sample_counts={idx: len(client.y_train) for idx, client in enumerate(clients)},
                batch_size=privacy_config.dp_batch_size,
                base_epochs=int(config.get("force_client_epochs") or args.client_epochs),
                model_bytes=parameter_bytes(global_params),
                participation_counts=participation_counts,
                contribution_scores=contribution_scores,
                quality_scores=quality_scores,
                privacy_states=client_privacy_state,
                remaining_rounds=args.num_rounds - round_num + 1,
                target_count=target_count,
                risk_actions=previous_risk_actions,
                eligible_indices=set(available_indices),
                residual_pressures=residual_pressures,
            )
            selected_indices = set(planned_actions)
            result["resource_trace_records"].extend(trace_rows)
            result["orchestration_decisions"].extend(
                [{"round": round_num, "client_id": clients[idx].client_id, "tier": resource_orchestrator.profiles[idx].tier,
                  **action.__dict__} for idx, action in planned_actions.items()]
            )
        else:
            if resource_simulation_enabled:
                fixed_epochs = int(config.get("force_client_epochs") or args.client_epochs)
                resource_available = []
                for idx in available_indices:
                    snapshot = resource_orchestrator.snapshot(idx, round_num)
                    _, _, total_seconds = resource_orchestrator.predict_seconds(
                        snapshot, len(clients[idx].y_train), privacy_config.dp_batch_size,
                        fixed_epochs, 1.0, parameter_bytes(global_params),
                    )
                    status = "eligible_static" if snapshot.online and total_seconds <= args.round_deadline_seconds else "deadline_or_offline"
                    result["resource_trace_records"].append({
                        **resource_orchestrator._trace_row(snapshot, None, status),
                        "fixed_epochs": fixed_epochs, "predicted_total_seconds": total_seconds,
                    })
                    if status == "eligible_static":
                        resource_available.append(idx)
                available_indices = resource_available
            selected_indices = set(
                _select_participants(
                    participation_policy,
                    available_indices,
                    clients,
                    capabilities,
                    contribution_scores,
                    participation_rate,
                    rng,
                )
            )
        for selected_idx in selected_indices:
            participation_counts[selected_idx] += 1

        for idx, client in enumerate(clients):
            if failure_plan[round_num - 1][idx] or idx not in selected_indices:
                round_update_norms.append(None)
                round_ebcd_stats.append((None, None, None))
                round_zkip_status.append(None)
                round_epsilons.append(client_privacy_state[idx]["accountant"].epsilon if idx in client_privacy_state else None)
                round_noise_multipliers.append(None)
                continue

            effective_epochs = _effective_epochs(idx, args.client_epochs, capabilities, config["compute_adapter"], privacy_config)
            if config.get("force_client_epochs") is not None:
                effective_epochs = int(config["force_client_epochs"])
            privacy_state = client_privacy_state.get(idx)
            noise_multiplier = None
            if arpa_enabled:
                action = planned_actions[idx]
                effective_epochs = action.epochs
                noise_multiplier = action.noise_multiplier
            elif privacy_state is not None:
                noise_multiplier = (
                    _apdp_noise_multiplier(idx, capabilities, participation_counts, privacy_state, privacy_config, round_num)
                    if config["dynamic_privacy"]
                    else privacy_state["base_noise_multiplier"]
                )
            round_noise_multipliers.append(noise_multiplier)
            client.set_global_model_parameters(global_params)
            original_X, original_y, polluted_sample_indices, polluted_sample_count = None, None, None, 0
            if idx in polluted_indices and _pollution_active(args, round_num):
                original_X, original_y, polluted_sample_indices, polluted_sample_count = _apply_client_pollution(
                    client,
                    classes,
                    args,
                    pollution_rng,
                )
                result["pollution_records"].append(
                    {
                        "round": round_num,
                        "client_id": client.client_id,
                        "pollution_type": args.pollution_type,
                        "polluted_sample_count": polluted_sample_count,
                        "pollution_rate": args.pollution_rate,
                    }
                )
            try:
                delta, proof = client.train(
                    epochs=effective_epochs,
                    use_dp=config["dp_scope"] == "client",
                    fedprox_mu=args.fedprox_mu if config["fedprox"] else 0.0,
                    global_params=global_params,
                    privacy_accountant=privacy_state["accountant"] if privacy_state is not None else None,
                    noise_multiplier=noise_multiplier,
                    round_num=round_num,
                )
            finally:
                _restore_client_data(client, original_X, original_y)
            if privacy_state is not None and client.last_privacy_event is not None:
                event = client.last_privacy_event
                result["privacy_accounting_records"].append({
                    "method": method_name, "client_id": client.client_id, "round": round_num,
                    "steps": event.steps, "sample_rate": event.sample_rate,
                    "noise_multiplier": event.noise_multiplier, "cumulative_epsilon": event.epsilon,
                    "incremental_epsilon": event.incremental_epsilon,
                    "target_epsilon": privacy_config.epsilon_per_client_total,
                    "remaining_epsilon": privacy_state["accountant"].remaining_epsilon, "status": event.status,
                })
            round_epsilons.append(privacy_state["accountant"].epsilon if privacy_state is not None else None)
            if delta is None:
                round_update_norms.append(None)
                round_ebcd_stats.append((None, None, None))
                round_zkip_status.append(False)
                continue

            if arpa_enabled:
                action = planned_actions[idx]
                residual_l2_before = float(np.sqrt(sum(np.linalg.norm(value) ** 2 for value in residuals[idx].values())))
                combined_delta = {key: delta[key] + residuals[idx][key] for key in delta}
                combined_delta_l2 = float(np.sqrt(sum(np.linalg.norm(value) ** 2 for value in combined_delta.values())))
                masks = rotating_block_mask(combined_delta, args.parameter_blocks, action.upload_ratio, idx, round_num)
                delta = mask_delta(combined_delta, masks)
                residuals[idx] = {key: combined_delta[key] - delta[key] for key in combined_delta}
                residual_l2_after = float(np.sqrt(sum(np.linalg.norm(value) ** 2 for value in residuals[idx].values())))
                proof = client.zkip.generate_proof(delta)
                masks_by_client[client.client_id] = masks
                uploaded_parameters = sum(int(np.count_nonzero(masks[key])) for key in delta)
                total_parameters = sum(int(np.asarray(delta[key]).size) for key in delta)
                uploaded_bytes = sum(int(np.count_nonzero(masks[key])) * np.asarray(delta[key]).dtype.itemsize for key in delta)
                result["partial_update_records"].append({
                    "round": round_num, "client_id": client.client_id, "upload_ratio": action.upload_ratio,
                    "parameter_blocks": args.parameter_blocks, "uploaded_bytes": uploaded_bytes,
                    "total_parameter_bytes": parameter_bytes(global_params),
                    "uploaded_parameters": uploaded_parameters,
                    "total_parameters": total_parameters,
                    "uploaded_parameter_fraction": uploaded_parameters / max(1, total_parameters),
                    "deadline_slack_ratio": action.deadline_slack_ratio,
                    "upload_selection_reason": action.upload_selection_reason,
                    "residual_pressure": action.residual_pressure,
                    "residual_l2_before": residual_l2_before,
                    "combined_delta_l2": combined_delta_l2,
                    "residual_l2": residual_l2_after,
                    "residual_l2_after": residual_l2_after,
                    "residual_feedback_full_upload": action.upload_selection_reason == "residual_feedback_full_upload",
                })

            update_norm = np.sqrt(sum(np.linalg.norm(v.flatten()) ** 2 for v in delta.values()))
            round_update_norms.append(update_norm)
            round_delta_norm += update_norm
            flat = _flatten_parameter_dict(delta)
            if flat.size:
                round_ebcd_stats.append((np.var(flat), kurtosis(flat, fisher=True), skew(flat)))
            else:
                round_ebcd_stats.append((None, None, None))
            zkip_ok = server.zkip.verify_proof(delta, proof) if config["use_zkip"] else True
            round_zkip_status.append(zkip_ok)
            local_steps = _local_step_count(len(client.y_train), effective_epochs, privacy_config.dp_batch_size)
            client_updates.append((delta, proof, client.client_id, len(client.y_train), len(client.y_train), local_steps))
            if noise_multiplier is not None:
                noise_stddev = noise_multiplier * client.dp_l2_norm_clip / max(1, min(privacy_config.dp_batch_size, len(client.y_train)))
            else:
                noise_stddev = 0.0
            round_noise_scales.append(noise_stddev)

        regulatory_records = []
        adjusted_weights = {}
        regulatory_zkip_failures = 0
        if regulatory_controller is not None:
            regulatory_records, adjusted_weights = regulatory_controller.evaluate_round(
                round_num,
                clients,
                selected_indices,
                round_update_norms,
                round_ebcd_stats,
                round_zkip_status,
            )
            regulatory_zkip_failures = sum(1 for row in regulatory_records if row["zkip_status"] is False)
            weighted_updates = []
            for update in client_updates:
                delta, proof, client_id, data_size = update[:4]
                local_steps = update[5] if len(update) >= 6 else 1
                adjusted_weight = adjusted_weights.get(client_id, data_size)
                if adjusted_weight > 0:
                    weighted_updates.append((delta, proof, client_id, data_size, adjusted_weight, local_steps))
            client_updates = weighted_updates
            previous_risk_actions = {
                int(str(row["client_id"]).split("_")[-1]): row["action"]
                for row in regulatory_records
            }
        else:
            adjusted_weights = {
                update[2]: update[3]
                for update in client_updates
            }

        if arpa_enabled:
            server.global_model_parameters, aggregation_success, aggregated_from, zkip_failures, server_noise_scale = _aggregate_masked_deltas(
                server.global_model_parameters, client_updates, masks_by_client, config["use_zkip"], server.zkip,
                privacy_config=privacy_config, apply_server_dp=config["dp_scope"] == "server",
            )
        else:
            server.global_model_parameters, aggregation_success, aggregated_from, zkip_failures, server_noise_scale = _aggregate_deltas(
                server.global_model_parameters, client_updates, config["use_zkip"], server.zkip,
                privacy_config=privacy_config, apply_server_dp=config["dp_scope"] == "server",
                config=config, optimizer_state=server_optimizer_state,
            )
        zkip_failures += regulatory_zkip_failures
        ebcd_alert = 1 if config["use_ebcd"] and server.ebcd.check_for_corruption(server.global_model_parameters) else 0
        if config["use_tcm"]:
            state_details = {
                "method": method_name,
                "aggregation_successful": aggregation_success,
                "aggregated_from_clients_count": len(aggregated_from),
                "arpa_enabled": arpa_enabled,
                "backend": backend,
                "device": str(backend_device) if backend_device is not None else None,
                "resource_deadline_seconds": args.round_deadline_seconds if arpa_enabled else None,
                "partial_update_clients": len(masks_by_client),
            }
            server.tcm.record_state(
                round_num,
                server.global_model_parameters,
                state_details,
                {cid: "OK" for cid in aggregated_from},
            )

        metrics = _evaluate_global_params(
            method_name,
            client_ids,
            num_features,
            classes,
            server.global_model_parameters,
            X_test,
            y_test,
            round_num,
            backend,
        )
        duration = time.time() - start_time
        valid_norms = [n for n in round_update_norms if n is not None]
        avg_delta_norm = round_delta_norm / len(valid_norms) if valid_norms else 0.0

        result["rounds"].append(round_num)
        result["accuracies"].append(_metric_value(metrics.get("accuracy")))
        result["balanced_accuracies"].append(_metric_value(metrics.get("balanced_accuracy")))
        result["f1_scores"].append(_metric_value(metrics.get("f1_score")))
        result["precisions"].append(_metric_value(metrics.get("precision")))
        result["recalls"].append(_metric_value(metrics.get("recall")))
        result["aucs"].append(_metric_value(metrics.get("auc_roc")))
        result["round_durations"].append(duration)
        result["agg_client_counts"].append(len(aggregated_from))
        result["selected_client_counts"].append(len(selected_indices))
        result["dp_noise_scales"].append(
            server_noise_scale if config["dp_scope"] == "server"
            else float(np.mean(round_noise_scales)) if round_noise_scales else 0.0
        )
        result["zkip_failures"].append(zkip_failures)
        result["delta_norms"].append(avg_delta_norm)
        result["ebcd_alerts"].append(ebcd_alert)
        result["tcm_counts"].append(len(server.tcm.manifold_log) if config["use_tcm"] else 0)
        result["per_client_update_norms"].append(round_update_norms)
        result["per_client_ebcd_stats"].append(round_ebcd_stats)
        result["per_client_zkip_status"].append(round_zkip_status)
        result["per_client_epsilon"].append(round_epsilons)
        result["per_client_noise_multiplier"].append(round_noise_multipliers)
        fairness_records = []
        if result["fairness_enabled"]:
            fairness_records, fairness_summary = _client_fairness_records(
                round_num,
                clients,
                server.global_model_parameters,
                classes,
                num_features,
                round_epsilons,
                selected_indices,
                participation_counts,
            )
            result["fairness_records"].extend(fairness_records)
            result["client_accuracy_gaps"].append(fairness_summary["client_accuracy_gap"])
            result["client_accuracy_stds"].append(fairness_summary["client_accuracy_std"])
            result["client_min_accuracies"].append(fairness_summary["client_min_accuracy"])
            result["epsilon_gaps"].append(fairness_summary["epsilon_gap"])
            result["epsilon_stds"].append(fairness_summary["epsilon_std"])
            result["participation_gaps"].append(fairness_summary["participation_gap"])
            result["participation_stds"].append(fairness_summary["participation_std"])
        else:
            result["client_accuracy_gaps"].append(0.0)
            result["client_accuracy_stds"].append(0.0)
            result["client_min_accuracies"].append(np.nan)
            result["epsilon_gaps"].append(0.0)
            result["epsilon_stds"].append(0.0)
            result["participation_gaps"].append(0.0)
            result["participation_stds"].append(0.0)
        if regulatory_controller is not None:
            result["regulatory_records"].extend(regulatory_records)
            result["regulatory_warning_counts"].append(sum(1 for row in regulatory_records if row["action"] == "warning"))
            result["regulatory_downweight_counts"].append(sum(1 for row in regulatory_records if row["action"] == "downweight"))
            result["regulatory_quarantine_counts"].append(sum(1 for row in regulatory_records if row["action"] == "quarantine"))
            risks = [row["risk_score"] for row in regulatory_records if np.isfinite(row["risk_score"])]
            result["regulatory_avg_risks"].append(float(np.mean(risks)) if risks else 0.0)
        else:
            result["regulatory_warning_counts"].append(0)
            result["regulatory_downweight_counts"].append(0)
            result["regulatory_quarantine_counts"].append(0)
            result["regulatory_avg_risks"].append(0.0)
        if contribution_evaluator is not None:
            contribution_records, contribution_summary = contribution_evaluator.evaluate_round(
                round_num,
                clients,
                selected_indices,
                global_params,
                client_updates,
                adjusted_weights,
                regulatory_records,
                fairness_records,
                round_update_norms,
                X_test,
                y_test,
            )
            result["contribution_records"].extend(contribution_records)
            result["avg_contribution_scores"].append(contribution_summary["avg_contribution_score"])
            result["contribution_score_gaps"].append(contribution_summary["contribution_score_gap"])
            result["contribution_score_stds"].append(contribution_summary["contribution_score_std"])
            result["avg_approx_shapley"].append(contribution_summary["avg_approx_shapley"])
            result["approx_shapley_gaps"].append(contribution_summary["approx_shapley_gap"])
            result["approx_shapley_stds"].append(contribution_summary["approx_shapley_std"])
            result["avg_risk_penalties"].append(contribution_summary["avg_risk_penalty"])
            result["avg_fairness_penalties"].append(contribution_summary["avg_fairness_penalty"])
            result["avg_weight_alignment_errors"].append(contribution_summary["avg_weight_alignment_error"])
            result["max_weight_alignment_errors"].append(contribution_summary["max_weight_alignment_error"])
        else:
            result["avg_contribution_scores"].append(0.0)
            result["contribution_score_gaps"].append(0.0)
            result["contribution_score_stds"].append(0.0)
            result["avg_approx_shapley"].append(0.0)
            result["approx_shapley_gaps"].append(0.0)
            result["approx_shapley_stds"].append(0.0)
            result["avg_risk_penalties"].append(0.0)
            result["avg_fairness_penalties"].append(0.0)
            result["avg_weight_alignment_errors"].append(0.0)
            result["max_weight_alignment_errors"].append(0.0)

        contribution_history.append(
            {
                idx: (
                    round_update_norms[idx] if idx < len(round_update_norms) and round_update_norms[idx] is not None else 0.0,
                    getattr(clients[idx], "last_val_acc_gain", 0.0),
                    getattr(clients[idx], "last_val_loss_drop", 0.0),
                )
                for idx in range(len(clients))
            }
        )
        contribution_scores = {
            idx: (
                round_update_norms[idx]
                if idx < len(round_update_norms) and round_update_norms[idx] is not None
                else contribution_scores.get(idx, 0.0)
            )
            for idx in range(len(clients))
        }
        print(
            f"{config['label']} round {round_num}/{args.num_rounds}: "
            f"Acc={result['accuracies'][-1]:.3f}, F1={result['f1_scores'][-1]:.3f}, "
            f"BalancedAcc={result['balanced_accuracies'][-1]:.3f}"
        )

    _save_method_artifacts(output_dir, method_name, result, client_ids)
    return result


def _parse_methods(methods_arg):
    if methods_arg == "all":
        return list(BASELINE_METHODS)
    methods = [m.strip().lower() for m in methods_arg.split(",") if m.strip()]
    invalid = [m for m in methods if m not in SUPPORTED_BASELINE_METHODS]
    if invalid:
        raise ValueError(f"Unsupported baseline methods: {invalid}. Supported: {SUPPORTED_BASELINE_METHODS}")
    return methods


def _save_baseline_method_metadata(output_dir, method_names):
    """Persist the exact comparison configuration used by a baseline run."""
    rows = []
    for method_name in method_names:
        config = METHOD_CONFIGS[method_name]
        rows.append(
            {
                "method": method_name,
                "label": config["label"],
                "comparison_role": "default_dp_baseline" if method_name in BASELINE_METHODS else "legacy_explicit_only",
                "dp_scope": config["dp_scope"],
                "use_dp": config["use_dp"],
                "dynamic_privacy": config["dynamic_privacy"],
                "compute_adapter": config["compute_adapter"],
                "use_zkip": config["use_zkip"],
                "use_ebcd": config["use_ebcd"],
                "use_tcm": config["use_tcm"],
                "fedprox": config["fedprox"],
                "force_client_epochs": config["force_client_epochs"],
                "reference": config["reference"],
            }
        )
    _write_csv(os.path.join(output_dir, "baseline_method_metadata.csv"), rows)


def _final_metric_row(method_name, result):
    row = {
        "method": method_name,
        "label": result["label"],
        "reference": result.get("reference", ""),
        "final_accuracy": result["accuracies"][-1] if result["accuracies"] else np.nan,
        "best_accuracy": np.nanmax(result["accuracies"]) if result["accuracies"] else np.nan,
        "final_balanced_accuracy": result["balanced_accuracies"][-1] if result["balanced_accuracies"] else np.nan,
        "best_balanced_accuracy": np.nanmax(result["balanced_accuracies"]) if result["balanced_accuracies"] else np.nan,
        "final_f1_score": result["f1_scores"][-1] if result["f1_scores"] else np.nan,
        "best_f1_score": np.nanmax(result["f1_scores"]) if result["f1_scores"] else np.nan,
        "final_auc_roc": result["aucs"][-1] if result["aucs"] else np.nan,
        "avg_round_time": np.nanmean(result["round_durations"]) if result["round_durations"] else np.nan,
        "avg_dp_noise_scale": np.nanmean(result["dp_noise_scales"]) if result["dp_noise_scales"] else np.nan,
        "avg_selected_client_count": np.nanmean(result.get("selected_client_counts", [])) if result.get("selected_client_counts") else np.nan,
        "avg_agg_client_count": np.nanmean(result["agg_client_counts"]) if result["agg_client_counts"] else np.nan,
        "participation_policy": result.get("participation_policy", ""),
        "total_zkip_failures": np.nansum(result["zkip_failures"]) if result["zkip_failures"] else 0,
        "total_ebcd_alerts": np.nansum(result["ebcd_alerts"]) if result["ebcd_alerts"] else 0,
        "final_tcm_state_count": result["tcm_counts"][-1] if result["tcm_counts"] else 0,
        "total_warnings": np.nansum(result.get("regulatory_warning_counts", [])) if result.get("regulatory_warning_counts") else 0,
        "total_downweighted": np.nansum(result.get("regulatory_downweight_counts", [])) if result.get("regulatory_downweight_counts") else 0,
        "total_quarantined": np.nansum(result.get("regulatory_quarantine_counts", [])) if result.get("regulatory_quarantine_counts") else 0,
        "avg_regulatory_risk": np.nanmean(result.get("regulatory_avg_risks", [])) if result.get("regulatory_avg_risks") else 0.0,
        "final_client_accuracy_gap": result["client_accuracy_gaps"][-1] if result.get("client_accuracy_gaps") else 0.0,
        "avg_client_accuracy_gap": np.nanmean(result.get("client_accuracy_gaps", [])) if result.get("client_accuracy_gaps") else 0.0,
        "final_client_accuracy_std": result["client_accuracy_stds"][-1] if result.get("client_accuracy_stds") else 0.0,
        "final_client_min_accuracy": result["client_min_accuracies"][-1] if result.get("client_min_accuracies") else np.nan,
        "final_epsilon_gap": result["epsilon_gaps"][-1] if result.get("epsilon_gaps") else 0.0,
        "avg_epsilon_gap": np.nanmean(result.get("epsilon_gaps", [])) if result.get("epsilon_gaps") else 0.0,
        "final_participation_gap": result["participation_gaps"][-1] if result.get("participation_gaps") else 0.0,
        "avg_participation_gap": np.nanmean(result.get("participation_gaps", [])) if result.get("participation_gaps") else 0.0,
        "final_group_accuracy_gap": result.get("final_group_accuracy_gap", 0.0),
        "final_worst_group_accuracy": result.get("final_worst_group_accuracy", np.nan),
        "final_group_f1_gap": result.get("final_group_f1_gap", 0.0),
        "final_group_epsilon_gap": result.get("final_group_epsilon_gap", 0.0),
        "final_group_participation_gap": result.get("final_group_participation_gap", 0.0),
        "final_group_quarantine_gap": result.get("final_group_quarantine_gap", 0.0),
        "avg_contribution_score": np.nanmean(result.get("avg_contribution_scores", [])) if result.get("avg_contribution_scores") else 0.0,
        "final_contribution_score_gap": result["contribution_score_gaps"][-1] if result.get("contribution_score_gaps") else 0.0,
        "avg_approx_shapley": np.nanmean(result.get("avg_approx_shapley", [])) if result.get("avg_approx_shapley") else 0.0,
        "final_approx_shapley_gap": result["approx_shapley_gaps"][-1] if result.get("approx_shapley_gaps") else 0.0,
        "avg_risk_penalty": np.nanmean(result.get("avg_risk_penalties", [])) if result.get("avg_risk_penalties") else 0.0,
        "avg_fairness_penalty": np.nanmean(result.get("avg_fairness_penalties", [])) if result.get("avg_fairness_penalties") else 0.0,
        "avg_weight_alignment_error": np.nanmean(result.get("avg_weight_alignment_errors", [])) if result.get("avg_weight_alignment_errors") else 0.0,
        "max_weight_alignment_error": np.nanmax(result.get("max_weight_alignment_errors", [])) if result.get("max_weight_alignment_errors") else 0.0,
    }
    row.update(_pollution_detection_metrics(result))
    return row


def _save_suite_summary(output_dir, method_results):
    summary_rows = []
    final_rows = []
    for method_name, result in method_results.items():
        final_rows.append(_final_metric_row(method_name, result))
        for idx, round_num in enumerate(result["rounds"]):
            summary_rows.append(
                {
                    "method": method_name,
                    "label": result["label"],
                    "round": round_num,
                    "accuracy": result["accuracies"][idx],
                    "balanced_accuracy": result["balanced_accuracies"][idx],
                    "f1_score": result["f1_scores"][idx],
                    "auc_roc": result["aucs"][idx],
                    "dp_noise_scale": result["dp_noise_scales"][idx],
                    "agg_client_count": result["agg_client_counts"][idx],
                    "selected_client_count": result.get("selected_client_counts", [np.nan] * len(result["rounds"]))[idx],
                    "participation_policy": result.get("participation_policy", ""),
                    "zkip_failures": result["zkip_failures"][idx],
                    "ebcd_alert": result["ebcd_alerts"][idx],
                    "tcm_state_count": result["tcm_counts"][idx],
                    "round_duration": result["round_durations"][idx],
                }
            )

    for filename, rows in (("baseline_summary.csv", summary_rows), ("baseline_final_metrics.csv", final_rows)):
        with open(os.path.join(output_dir, filename), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["method"])
            writer.writeheader()
            writer.writerows(rows)

    regulatory_rows = []
    for method_name, result in method_results.items():
        for row in result.get("regulatory_records", []):
            regulatory_rows.append({"method": method_name, **row})
    if regulatory_rows:
        _write_csv(os.path.join(output_dir, "regulatory_intervention_summary.csv"), regulatory_rows)

        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        for action_key, label in (
            ("regulatory_warning_counts", "warning"),
            ("regulatory_downweight_counts", "downweight"),
            ("regulatory_quarantine_counts", "quarantine"),
        ):
            totals = []
            rounds = next(iter(method_results.values()))["rounds"] if method_results else []
            for idx, _ in enumerate(rounds):
                totals.append(sum(result.get(action_key, [0] * len(rounds))[idx] for result in method_results.values()))
            plt.plot(rounds, totals, marker="o", label=label)
        plt.xlabel("Round")
        plt.ylabel("Client Count")
        plt.title("Regulatory Intervention Actions")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "regulatory_actions.png"))
        plt.close()

        plt.figure(figsize=charts.FIGSIZE_WIDE)
        client_keys = sorted({(row["method"], row["client_id"]) for row in regulatory_rows})
        for method_name, client_id in client_keys:
            rows = [
                row for row in regulatory_rows
                if row["method"] == method_name and row["client_id"] == client_id
            ]
            plt.plot([row["round"] for row in rows], [row["risk_score"] for row in rows], marker="o", label=f"{method_name}:{client_id}")
        plt.xlabel("Round")
        plt.ylabel("Risk Score")
        plt.title("Regulatory Risk by Client")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "regulatory_risk_by_client.png"))
        plt.close()

    fairness_rows = []
    for method_name, result in method_results.items():
        for row in result.get("fairness_records", []):
            fairness_rows.append({"method": method_name, **row})
    if fairness_rows:
        _write_csv(os.path.join(output_dir, "client_fairness_summary.csv"), fairness_rows)

        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        for method_name, result in method_results.items():
            if result.get("client_accuracy_gaps"):
                plt.plot(result["rounds"], result["client_accuracy_gaps"], marker="o", label=method_name)
        plt.xlabel("Round")
        plt.ylabel("Client Accuracy Gap")
        plt.title("Client Performance Fairness")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "client_performance_fairness.png"))
        plt.close()

        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        for method_name, result in method_results.items():
            if result.get("epsilon_gaps"):
                plt.plot(result["rounds"], result["epsilon_gaps"], marker="o", label=method_name)
        plt.xlabel("Round")
        plt.ylabel("Privacy Budget Gap")
        plt.title("Privacy Budget Fairness")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "privacy_budget_fairness.png"))
        plt.close()

        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        for method_name, result in method_results.items():
            if result.get("participation_gaps"):
                plt.plot(result["rounds"], result["participation_gaps"], marker="o", label=method_name)
        plt.xlabel("Round")
        plt.ylabel("Participation Count Gap")
        plt.title("Client Participation Fairness")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "participation_fairness.png"))
        plt.close()

    contribution_rows = []
    for method_name, result in method_results.items():
        for row in result.get("contribution_records", []):
            contribution_rows.append({"method": method_name, **row})
    if contribution_rows:
        _write_csv(os.path.join(output_dir, "contribution_penalty_summary.csv"), contribution_rows)

        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        for method_name, result in method_results.items():
            if result.get("avg_contribution_scores"):
                plt.plot(result["rounds"], result["avg_contribution_scores"], marker="o", label=method_name)
        plt.xlabel("Round")
        plt.ylabel("Average Contribution Score")
        plt.title("Average Contribution Score")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "contribution_score_by_client.png"))
        plt.close()

        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        for method_name, result in method_results.items():
            if result.get("avg_approx_shapley"):
                plt.plot(result["rounds"], result["avg_approx_shapley"], marker="o", label=method_name)
        plt.xlabel("Round")
        plt.ylabel("Average Approximate Shapley")
        plt.title("Average Leave-One-Out Approximate Shapley")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "approx_shapley_by_client.png"))
        plt.close()

        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        for method_name, result in method_results.items():
            if result.get("avg_risk_penalties"):
                plt.plot(result["rounds"], result["avg_risk_penalties"], marker="o", label=f"{method_name} risk")
            if result.get("avg_fairness_penalties"):
                plt.plot(result["rounds"], result["avg_fairness_penalties"], marker="x", label=f"{method_name} fairness")
        plt.xlabel("Round")
        plt.ylabel("Average Penalty")
        plt.title("Penalty Components")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "penalty_components.png"))
        plt.close()

        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        for method_name, result in method_results.items():
            if result.get("avg_weight_alignment_errors"):
                plt.plot(result["rounds"], result["avg_weight_alignment_errors"], marker="o", label=f"{method_name} avg")
            if result.get("max_weight_alignment_errors"):
                plt.plot(result["rounds"], result["max_weight_alignment_errors"], marker="x", label=f"{method_name} max")
        plt.xlabel("Round")
        plt.ylabel("Alignment Error")
        plt.title("Contribution-Weight Alignment")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "contribution_weight_alignment.png"))
        plt.close()

    plt.figure(figsize=charts.FIGSIZE_COMPARISON)
    for metric_idx, (metric_key, title) in enumerate(
        (("accuracies", "Accuracy"), ("f1_scores", "Macro-F1"), ("balanced_accuracies", "Balanced Accuracy")),
        start=1,
    ):
        plt.subplot(3, 1, metric_idx)
        for method_name, result in method_results.items():
            plt.plot(result["rounds"], result[metric_key], label=result["label"])
        plt.ylabel(title)
        plt.grid(True, alpha=0.3)
        if metric_idx == 1:
            plt.legend(loc="best")
        if metric_idx == 3:
            plt.xlabel("Round")
    plt.tight_layout()
    charts.save_figure(os.path.join(output_dir, "baseline_comparison.png"))
    plt.close()
    write_artifact_manifest(output_dir)


def _load_suite_data(args, privacy_config):
    X_train_full, y_train_full, X_test, y_test, _, classes, presplit_client_data = load_experiment_data(
        dataset_name=args.dataset,
        data_root=args.data_root,
        random_state=args.seed,
        max_samples=args.max_samples,
        emnist_split=args.emnist_split,
    )
    if X_train_full is None:
        raise RuntimeError("Failed to load data.")
    if presplit_client_data is not None:
        client_datasets = presplit_client_data
        args.num_clients = len(client_datasets)
    else:
        client_datasets = split_data_for_clients(
            X_train_full,
            y_train_full,
            args.num_clients,
            size_ratios=None,
            partition=args.partition,
            dirichlet_alpha=args.dirichlet_alpha,
            random_state=args.seed,
        )

    train_val_data = _split_train_val(client_datasets, classes, args.seed)
    rng = np.random.default_rng(args.seed + 2026)
    failure_plan = rng.random((args.num_rounds, args.num_clients)) < privacy_config.failure_prob
    return train_val_data, X_test, y_test, classes, failure_plan


def _synthetic_capabilities(metadata_rows):
    value_by_level = {"high": 1.0, "medium": 0.65, "low": 0.30}
    return {row["client_idx"]: value_by_level.get(row["compute_level"], 0.65) for row in metadata_rows}


def _synthetic_failure_plan(args, metadata_rows, privacy_config):
    rng = np.random.default_rng(args.seed + 3030)
    region_prob = {"east": 0.05, "central": 0.15, "west": 0.35}
    quality_bonus = {"clean": 0.0, "biased": 0.03, "noisy": 0.08}
    plan = np.zeros((args.num_rounds, args.num_clients), dtype=bool)
    for row in metadata_rows:
        idx = row["client_idx"]
        prob = region_prob.get(row["region_group"], privacy_config.failure_prob)
        prob += quality_bonus.get(row["data_quality"], 0.0)
        prob = min(0.85, max(0.0, prob))
        plan[:, idx] = rng.random(args.num_rounds) < prob
    return plan


def _load_synthetic_fairness_data(args, privacy_config, dataset_name):
    data_args = copy.copy(args)
    data_args.dataset = dataset_name
    X_train_full, y_train_full, X_test, y_test, _, classes, presplit_client_data = load_experiment_data(
        dataset_name=dataset_name,
        data_root=args.data_root,
        random_state=args.seed,
        max_samples=args.max_samples,
        emnist_split=args.emnist_split,
    )
    if X_train_full is None:
        raise RuntimeError(f"Failed to load data for {dataset_name}.")
    if presplit_client_data is not None:
        client_datasets = presplit_client_data
        data_args.num_clients = len(client_datasets)
    else:
        client_datasets = split_data_for_clients(
            X_train_full,
            y_train_full,
            args.num_clients,
            size_ratios=None,
            partition=args.partition,
            dirichlet_alpha=args.dirichlet_alpha,
            random_state=args.seed,
        )

    assigner = SyntheticSensitiveAttributeAssigner(args.synthetic_sensitive_attrs, args.fairness_pressure_profile)
    metadata_rows = assigner.assign(client_datasets)
    applier = SyntheticFairnessPressureApplier(seed=args.seed + 4040)
    pressured_datasets, metadata_rows = applier.apply(client_datasets, metadata_rows)
    data_args.num_clients = len(pressured_datasets)
    train_val_data = _split_train_val(pressured_datasets, classes, args.seed)
    capabilities = _synthetic_capabilities(metadata_rows)
    failure_plan = _synthetic_failure_plan(data_args, metadata_rows, privacy_config)
    for row in metadata_rows:
        row["dataset"] = dataset_name
    return data_args, train_val_data, X_test, y_test, classes, failure_plan, metadata_rows, capabilities


def _write_csv(path, rows):
    if rows:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    else:
        fieldnames = ["name"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _present_for_audit(value):
    if value is None:
        return False
    if value == "":
        return False
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def _audit_hash(payload, previous_hash, algorithm="sha256"):
    if algorithm != "sha256":
        raise ValueError(f"Unsupported audit digest algorithm: {algorithm}")
    safe_payload = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest_input = f"{previous_hash}|{safe_payload}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def _build_audit_trace(method_name, result, digest_algorithm="sha256"):
    regulatory_by_key = {(row["round"], row["client_id"]): row for row in result.get("regulatory_records", [])}
    fairness_by_key = {(row["round"], row["client_id"]): row for row in result.get("fairness_records", [])}
    contribution_by_key = {(row["round"], row["client_id"]): row for row in result.get("contribution_records", [])}
    client_count = 0
    if result.get("per_client_update_norms"):
        client_count = len(result["per_client_update_norms"][0])
    client_ids = [f"client_{idx}" for idx in range(client_count)]

    rows = []
    previous_hash = "GENESIS"
    for round_idx, round_num in enumerate(result.get("rounds", [])):
        update_norms = result.get("per_client_update_norms", [])[round_idx]
        ebcd_stats = result.get("per_client_ebcd_stats", [])[round_idx]
        zkip_status = result.get("per_client_zkip_status", [])[round_idx]
        epsilons = result.get("per_client_epsilon", [])[round_idx]
        for client_idx, client_id in enumerate(client_ids):
            key = (round_num, client_id)
            regulatory = regulatory_by_key.get(key, {})
            fairness = fairness_by_key.get(key, {})
            contribution = contribution_by_key.get(key, {})
            stat_values = ebcd_stats[client_idx] if client_idx < len(ebcd_stats) else (None, None, None)
            payload = {
                "method": method_name,
                "round": round_num,
                "client_id": client_id,
                "selected": fairness.get("selected", contribution.get("selected", client_idx < len(update_norms) and update_norms[client_idx] is not None)),
                "participation_count": fairness.get("participation_count", np.nan),
                "epsilon": epsilons[client_idx] if client_idx < len(epsilons) else None,
                "update_norm": update_norms[client_idx] if client_idx < len(update_norms) else None,
                "zkip_status": zkip_status[client_idx] if client_idx < len(zkip_status) else None,
                "ebcd_variance": stat_values[0] if stat_values is not None else None,
                "ebcd_kurtosis": stat_values[1] if stat_values is not None else None,
                "ebcd_skewness": stat_values[2] if stat_values is not None else None,
                "regulatory_action": regulatory.get("action", "not_evaluated"),
                "risk_score": regulatory.get("risk_score", np.nan),
                "adjusted_weight": regulatory.get("adjusted_weight", contribution.get("adjusted_weight", np.nan)),
                "local_accuracy": fairness.get("local_accuracy", np.nan),
                "local_f1_score": fairness.get("local_f1_score", np.nan),
                "data_quality_score": contribution.get("data_quality_score", np.nan),
                "approx_shapley": contribution.get("approx_shapley", np.nan),
                "final_contribution_score": contribution.get("final_contribution_score", np.nan),
                "contribution_weight_alignment_error": contribution.get("contribution_weight_alignment_error", np.nan),
            }
            completeness_fields = [
                "selected",
                "epsilon",
                "update_norm",
                "zkip_status",
                "regulatory_action",
                "risk_score",
                "adjusted_weight",
                "local_accuracy",
                "data_quality_score",
                "approx_shapley",
                "final_contribution_score",
            ]
            payload["trace_completeness"] = sum(
                1 for field in completeness_fields if _present_for_audit(payload.get(field))
            ) / len(completeness_fields)
            event_hash = _audit_hash(payload, previous_hash, digest_algorithm)
            rows.append(
                {
                    "audit_event_id": len(rows) + 1,
                    "previous_hash": previous_hash,
                    "event_hash": event_hash,
                    **payload,
                }
            )
            previous_hash = event_hash
    return rows


def _verify_audit_trace(rows, digest_algorithm="sha256"):
    verification_rows = []
    previous_hash = "GENESIS"
    for row in rows:
        payload = {
            key: value
            for key, value in row.items()
            if key not in {"audit_event_id", "previous_hash", "event_hash", "hash_valid", "previous_hash_valid"}
        }
        previous_hash_valid = row["previous_hash"] == previous_hash
        expected_hash = _audit_hash(payload, row["previous_hash"], digest_algorithm)
        hash_valid = row["event_hash"] == expected_hash
        verification_rows.append(
            {
                "audit_event_id": row["audit_event_id"],
                "method": row["method"],
                "round": row["round"],
                "client_id": row["client_id"],
                "previous_hash_valid": previous_hash_valid,
                "hash_valid": hash_valid,
                "expected_hash": expected_hash,
                "event_hash": row["event_hash"],
            }
        )
        previous_hash = row["event_hash"]
    return verification_rows


def _audit_summary_rows(method_results, audit_rows, verification_rows):
    rows = []
    for method_name, result in method_results.items():
        method_audit = [row for row in audit_rows if row["method"] == method_name]
        method_verify = [row for row in verification_rows if row["method"] == method_name]
        regulatory_rows = result.get("regulatory_records", [])
        rows.append(
            {
                "method": method_name,
                "total_audit_events": len(method_audit),
                "verified_audit_events": sum(1 for row in method_verify if row["hash_valid"] and row["previous_hash_valid"]),
                "invalid_chain_links": sum(1 for row in method_verify if not row["hash_valid"] or not row["previous_hash_valid"]),
                "avg_trace_completeness": float(np.mean([row["trace_completeness"] for row in method_audit])) if method_audit else 0.0,
                "total_warnings": sum(1 for row in regulatory_rows if row["action"] == "warning"),
                "total_downweighted": sum(1 for row in regulatory_rows if row["action"] == "downweight"),
                "total_quarantined": sum(1 for row in regulatory_rows if row["action"] == "quarantine"),
                "avg_weight_alignment_error": np.nanmean(result.get("avg_weight_alignment_errors", [])) if result.get("avg_weight_alignment_errors") else 0.0,
                "first_event_hash": method_audit[0]["event_hash"] if method_audit else "",
                "final_event_hash": method_audit[-1]["event_hash"] if method_audit else "",
            }
        )
    return rows


def _plot_audit_trace(audit_rows, output_dir):
    if not audit_rows:
        return
    plt.figure(figsize=charts.FIGSIZE_DEFAULT)
    for method_name in sorted({row["method"] for row in audit_rows}):
        method_rows = [row for row in audit_rows if row["method"] == method_name]
        rounds = sorted({row["round"] for row in method_rows})
        completeness = [
            float(np.mean([row["trace_completeness"] for row in method_rows if row["round"] == round_num]))
            for round_num in rounds
        ]
        plt.plot(rounds, completeness, marker="o", label=method_name)
    plt.xlabel("Round")
    plt.ylabel("Average Trace Completeness")
    plt.title("Audit Trace Completeness")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    charts.save_figure(os.path.join(output_dir, "audit_trace_timeline.png"))
    plt.close()


def _plot_group_metric(summary_rows, group_key, metric_key, output_path, ylabel, title):
    groups = {}
    for row in summary_rows:
        groups.setdefault(row[group_key], []).append(row)
    plt.figure(figsize=charts.FIGSIZE_DEFAULT)
    for group, rows in groups.items():
        rows = sorted(rows, key=lambda r: r["round"])
        plt.plot([r["round"] for r in rows], [r[metric_key] for r in rows], marker="o", label=str(group))
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    charts.save_figure(output_path)
    plt.close()


def _summary_rows(rows):
    return [row for row in rows if row.get("group") == "__summary__"]


def _last_round(rows):
    if not rows:
        return []
    last = max(row["round"] for row in rows)
    return [row for row in rows if row["round"] == last]


def _max_metric(rows, metric_key, default=0.0):
    values = [_metric_value(row.get(metric_key)) for row in rows]
    values = [value for value in values if np.isfinite(value)]
    return float(max(values)) if values else default


def _min_metric(rows, metric_key, default=np.nan):
    values = [_metric_value(row.get(metric_key)) for row in rows]
    values = [value for value in values if np.isfinite(value)]
    return float(min(values)) if values else default


def _group_gap(rows, metric_key):
    gaps = []
    for attr in sorted({row.get("attribute") for row in rows}):
        attr_rows = [row for row in rows if row.get("attribute") == attr]
        values = [_metric_value(row.get(metric_key)) for row in attr_rows]
        values = [value for value in values if np.isfinite(value)]
        if values:
            gaps.append(max(values) - min(values))
    return float(max(gaps)) if gaps else 0.0


def _apply_synthetic_final_metrics(result, group_rows, federated_rows):
    final_summary = _last_round(_summary_rows(group_rows))
    final_federated = _last_round(federated_rows)
    result["final_group_accuracy_gap"] = _max_metric(final_summary, "group_accuracy_gap")
    result["final_worst_group_accuracy"] = _min_metric(final_summary, "worst_group_accuracy")
    result["final_group_f1_gap"] = _max_metric(final_summary, "group_f1_gap")
    result["final_group_epsilon_gap"] = _group_gap(final_federated, "avg_epsilon")
    result["final_group_participation_gap"] = _group_gap(final_federated, "avg_participation_rounds")
    result["final_group_quarantine_gap"] = _group_gap(final_federated, "quarantine_count")


def _plot_synthetic_summary(group_rows, federated_rows, output_dir):
    summary = _summary_rows(group_rows)
    plot_specs = [
        (summary, "group_accuracy_gap", "group_accuracy_gap.png", "Accuracy Gap", "Group Accuracy Gap"),
        (summary, "worst_group_accuracy", "worst_group_accuracy.png", "Worst-group Accuracy", "Worst-group Accuracy"),
        (summary, "group_f1_gap", "group_macro_f1_gap.png", "Macro-F1 Gap", "Group Macro-F1 Gap"),
        (federated_rows, "avg_epsilon", "group_privacy_budget_gap.png", "Average Epsilon", "Group Privacy Budget"),
        (federated_rows, "avg_participation_rounds", "group_participation_gap.png", "Avg Participation Rounds", "Group Participation"),
        (federated_rows, "quarantine_count", "group_regulatory_actions.png", "Quarantine Count", "Group Regulatory Actions"),
    ]
    for rows, metric_key, filename, ylabel, title in plot_specs:
        if not rows:
            continue
        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        groups = {}
        for row in rows:
            key = f"{row.get('method')}:{row.get('attribute')}:{row.get('group')}"
            groups.setdefault(key, []).append(row)
        for key, key_rows in groups.items():
            key_rows = sorted(key_rows, key=lambda row: row["round"])
            values = [_metric_value(row.get(metric_key)) for row in key_rows]
            plt.plot([row["round"] for row in key_rows], values, marker="o", label=key)
        plt.xlabel("Round")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, filename))
        plt.close()


def _plot_final_metric(final_rows, x_key, metric_key, output_path, xlabel, ylabel, title):
    rows = sorted(final_rows, key=lambda r: r[x_key])
    plt.figure(figsize=charts.FIGSIZE_DEFAULT)
    plt.plot([r[x_key] for r in rows], [r[metric_key] for r in rows], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    charts.save_figure(output_path)
    plt.close()


def _plot_final_metric_by_method(final_rows, metric_key, output_path, ylabel, title):
    methods = sorted({row["method"] for row in final_rows})
    plt.figure(figsize=charts.FIGSIZE_DEFAULT)
    for method_name in methods:
        rows = sorted([row for row in final_rows if row["method"] == method_name], key=lambda r: r["budget"])
        plt.plot([row["budget"] for row in rows], [row[metric_key] for row in rows], marker="o", label=method_name)
    plt.xlabel("Total Privacy Budget")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    charts.save_figure(output_path)
    plt.close()


def _print_privacy_config(privacy_config):
    print(
        "Privacy config: "
        f"per_client_total_epsilon={privacy_config.epsilon_per_client_total}, "
        f"budget_semantics={privacy_config.budget_semantics}, "
        f"dp_batch_size={privacy_config.dp_batch_size}, "
        f"min_epsilon={privacy_config.min_epsilon}, "
        f"max_epsilon={privacy_config.max_epsilon}, "
        f"dp_l2_norm_clip={privacy_config.dp_l2_norm_clip}, "
        f"failure_prob={privacy_config.failure_prob}, "
        f"apdp_warmup_rounds={privacy_config.apdp_warmup_rounds}, "
        f"adaptive_increase_factor={privacy_config.adaptive_increase_factor}, "
        f"adaptive_decrease_factor={privacy_config.adaptive_decrease_factor}, "
        f"disable_compute_epoch_scaling={privacy_config.disable_compute_epoch_scaling}"
    )


def run_baseline_suite(args, output_dir):
    methods = _parse_methods(args.methods)
    privacy_config = make_privacy_config(args)
    print(f"Running baseline suite: {methods}")
    print(f"Results will be saved to: {output_dir}")
    _print_privacy_config(privacy_config)
    _save_baseline_method_metadata(output_dir, methods)

    train_val_data, X_test, y_test, classes, failure_plan = _load_suite_data(args, privacy_config)
    write_data_artifacts(output_dir, args, train_val_data, X_test, y_test, classes, failure_plan)

    method_results = {}
    for method_name in methods:
        method_output_dir = os.path.join(output_dir, method_name)
        method_results[method_name] = _run_single_method(
            method_name,
            args,
            train_val_data,
            X_test,
            y_test,
            classes,
            failure_plan,
            method_output_dir,
            privacy_config,
        )
    _save_suite_summary(output_dir, method_results)
    print(f"Baseline suite artifacts saved to: {output_dir}")


def run_pollution_injection_suite(args, output_dir):
    methods = _parse_csv_list(args.methods if args.methods != "all" else "grail_fl", POLLUTION_METHODS, "pollution methods")
    privacy_config = make_privacy_config(args)
    print(f"Running pollution injection suite: methods={methods}, type={args.pollution_type}, clients={args.polluted_clients}")
    print(f"Results will be saved to: {output_dir}")
    _print_privacy_config(privacy_config)

    train_val_data, X_test, y_test, classes, failure_plan = _load_suite_data(args, privacy_config)
    write_data_artifacts(output_dir, args, train_val_data, X_test, y_test, classes, failure_plan)
    scenario_results = {}
    final_rows = []
    summary_rows = []
    pollution_rows = []
    regulatory_rows = []

    for method_name in methods:
        scenarios = (
            ("pollution_no_intervention", False),
            ("pollution_with_intervention", True),
        )
        for scenario_name, enable_regulatory in scenarios:
            scenario_args = copy.copy(args)
            scenario_args.enable_pollution_injection = True
            scenario_args.enable_regulatory_intervention = enable_regulatory
            scenario_output_dir = os.path.join(output_dir, scenario_name, method_name)
            result = _run_single_method(
                method_name,
                scenario_args,
                train_val_data,
                X_test,
                y_test,
                classes,
                failure_plan,
                scenario_output_dir,
                privacy_config,
                label_suffix=f" ({scenario_name})",
            )
            scenario_key = f"{method_name}_{scenario_name}"
            scenario_results[scenario_key] = result
            final_row = _final_metric_row(scenario_key, result)
            final_row["method"] = method_name
            final_row["scenario"] = scenario_name
            final_rows.append(final_row)
            for idx, round_num in enumerate(result["rounds"]):
                summary_rows.append(
                    {
                        "method": method_name,
                        "scenario": scenario_name,
                        "round": round_num,
                        "accuracy": result["accuracies"][idx],
                        "balanced_accuracy": result["balanced_accuracies"][idx],
                        "f1_score": result["f1_scores"][idx],
                        "agg_client_count": result["agg_client_counts"][idx],
                        "total_regulatory_actions": (
                            result["regulatory_warning_counts"][idx]
                            + result["regulatory_downweight_counts"][idx]
                            + result["regulatory_quarantine_counts"][idx]
                        ),
                        "avg_regulatory_risk": result["regulatory_avg_risks"][idx],
                    }
                )
            for row in result.get("pollution_records", []):
                pollution_rows.append({"method": method_name, "scenario": scenario_name, **row})
            for row in result.get("regulatory_records", []):
                regulatory_rows.append({"method": method_name, "scenario": scenario_name, **row})

    _write_csv(os.path.join(output_dir, "pollution_summary.csv"), summary_rows)
    _write_csv(os.path.join(output_dir, "pollution_final_metrics.csv"), final_rows)
    _write_csv(os.path.join(output_dir, "pollution_injection_summary.csv"), pollution_rows)
    if regulatory_rows:
        _write_csv(os.path.join(output_dir, "regulatory_intervention_summary.csv"), regulatory_rows)

    _plot_group_metric(summary_rows, "scenario", "accuracy", os.path.join(output_dir, "pollution_accuracy.png"), "Accuracy", "Pollution Scenario Accuracy")
    _plot_group_metric(summary_rows, "scenario", "f1_score", os.path.join(output_dir, "pollution_f1.png"), "Macro-F1", "Pollution Scenario Macro-F1")
    _plot_group_metric(summary_rows, "scenario", "total_regulatory_actions", os.path.join(output_dir, "pollution_regulatory_actions.png"), "Regulatory Actions", "Regulatory Actions under Pollution")

    plt.figure(figsize=charts.FIGSIZE_DEFAULT)
    scenarios = [row["scenario"] for row in final_rows]
    detection_rates = [row["pollution_detection_rate"] for row in final_rows]
    plt.bar(scenarios, detection_rates)
    plt.ylabel("Detection Rate")
    plt.title("Pollution Detection Rate")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    charts.save_figure(os.path.join(output_dir, "pollution_detection_rate.png"))
    plt.close()
    write_artifact_manifest(output_dir)
    print(f"Pollution injection suite artifacts saved to: {output_dir}")


def run_fairness_suite(args, output_dir):
    methods = _parse_csv_list(args.fairness_methods, FAIRNESS_METHODS, "fairness methods")
    privacy_config = make_privacy_config(args)
    print(f"Running client fairness suite: methods={methods}")
    print(f"Results will be saved to: {output_dir}")
    _print_privacy_config(privacy_config)

    train_val_data, X_test, y_test, classes, failure_plan = _load_suite_data(args, privacy_config)
    write_data_artifacts(output_dir, args, train_val_data, X_test, y_test, classes, failure_plan)
    method_results = {}
    for method_name in methods:
        method_args = copy.copy(args)
        method_args.enable_fairness_evaluation = True
        method_output_dir = os.path.join(output_dir, method_name)
        method_results[method_name] = _run_single_method(
            method_name,
            method_args,
            train_val_data,
            X_test,
            y_test,
            classes,
            failure_plan,
            method_output_dir,
            privacy_config,
        )
    _save_suite_summary(output_dir, method_results)
    print(f"Client fairness suite artifacts saved to: {output_dir}")


def run_contribution_suite(args, output_dir):
    methods = _parse_csv_list(args.contribution_methods, CONTRIBUTION_METHODS, "contribution methods")
    privacy_config = make_privacy_config(args)
    print(f"Running penalty and Shapley contribution suite: methods={methods}")
    print(f"Results will be saved to: {output_dir}")
    _print_privacy_config(privacy_config)

    train_val_data, X_test, y_test, classes, failure_plan = _load_suite_data(args, privacy_config)
    write_data_artifacts(output_dir, args, train_val_data, X_test, y_test, classes, failure_plan)
    method_results = {}
    for method_name in methods:
        method_args = copy.copy(args)
        method_args.enable_contribution_evaluation = True
        method_args.enable_fairness_evaluation = True
        method_args.enable_regulatory_intervention = True
        method_output_dir = os.path.join(output_dir, method_name)
        method_results[method_name] = _run_single_method(
            method_name,
            method_args,
            train_val_data,
            X_test,
            y_test,
            classes,
            failure_plan,
            method_output_dir,
            privacy_config,
            label_suffix=" (contribution)",
        )
    _save_suite_summary(output_dir, method_results)
    print(f"Penalty and Shapley contribution suite artifacts saved to: {output_dir}")


def run_audit_trace_suite(args, output_dir):
    methods = _parse_csv_list(args.audit_methods, AUDIT_METHODS, "audit methods")
    privacy_config = make_privacy_config(args)
    print(f"Running audit trace suite: methods={methods}")
    print(f"Results will be saved to: {output_dir}")
    _print_privacy_config(privacy_config)

    train_val_data, X_test, y_test, classes, failure_plan = _load_suite_data(args, privacy_config)
    write_data_artifacts(output_dir, args, train_val_data, X_test, y_test, classes, failure_plan)
    method_results = {}
    audit_rows = []
    for method_name in methods:
        method_args = copy.copy(args)
        method_args.enable_regulatory_intervention = True
        method_args.enable_fairness_evaluation = True
        method_args.enable_contribution_evaluation = True
        method_output_dir = os.path.join(output_dir, method_name)
        result = _run_single_method(
            method_name,
            method_args,
            train_val_data,
            X_test,
            y_test,
            classes,
            failure_plan,
            method_output_dir,
            privacy_config,
            label_suffix=" (audit trace)",
        )
        method_results[method_name] = result
        audit_rows.extend(_build_audit_trace(method_name, result, args.audit_digest_algorithm))

    verification_rows = _verify_audit_trace(audit_rows, args.audit_digest_algorithm)
    summary_rows = _audit_summary_rows(method_results, audit_rows, verification_rows)
    _write_csv(os.path.join(output_dir, "audit_trace_log.csv"), audit_rows)
    _write_csv(os.path.join(output_dir, "audit_chain_verification.csv"), verification_rows)
    _write_csv(os.path.join(output_dir, "audit_trace_summary.csv"), summary_rows)
    _plot_audit_trace(audit_rows, output_dir)
    _save_suite_summary(output_dir, method_results)
    print(f"Audit trace suite artifacts saved to: {output_dir}")


def _ablation_config_overrides(scenario):
    if scenario == "no_adaptive_privacy":
        return {"dynamic_privacy": False}
    if scenario == "no_compute_adapter":
        return {"compute_adapter": False}
    if scenario == "no_resource_orchestration":
        return {"resource_orchestrator": False}
    if scenario == "no_partial_updates":
        return {"force_full_upload": True}
    if scenario == "no_resource_fairness":
        return {"resource_fairness": False}
    if scenario == "no_opportunity_privacy":
        return {"opportunity_privacy": False}
    if scenario == "no_budget_utilization_boost":
        return {"budget_utilization_boost": False}
    if scenario == "no_low_resource_compensation":
        return {"low_resource_compensation": False}
    if scenario == "no_zkip":
        return {"use_zkip": False}
    if scenario == "no_ebcd":
        return {"use_ebcd": False}
    if scenario == "no_tcm":
        return {"use_tcm": False}
    return {}


def _ablation_args(args, scenario):
    scenario_args = copy.copy(args)
    scenario_args.enable_regulatory_intervention = scenario != "no_regulatory"
    scenario_args.enable_contribution_evaluation = scenario != "no_contribution"
    scenario_args.enable_fairness_evaluation = scenario != "no_fairness"
    return scenario_args


def _plot_ablation_summary(summary_rows, final_rows, output_dir):
    if not summary_rows:
        return
    for metric_key, filename, ylabel, title in (
        ("accuracy", "ablation_accuracy.png", "Accuracy", "Ablation Accuracy"),
        ("f1_score", "ablation_macro_f1.png", "Macro-F1", "Ablation Macro-F1"),
        ("balanced_accuracy", "ablation_balanced_accuracy.png", "Balanced Accuracy", "Ablation Balanced Accuracy"),
    ):
        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        for scenario in sorted({row["scenario"] for row in summary_rows}):
            rows = sorted([row for row in summary_rows if row["scenario"] == scenario], key=lambda row: row["round"])
            plt.plot([row["round"] for row in rows], [row[metric_key] for row in rows], marker="o", label=scenario)
        plt.xlabel("Round")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, filename))
        plt.close()

    if final_rows:
        rows = sorted(final_rows, key=lambda row: row["scenario"])
        plt.figure(figsize=charts.FIGSIZE_WIDE)
        x = np.arange(len(rows))
        plt.bar(x, [row["final_accuracy_delta_vs_full"] for row in rows])
        plt.xticks(x, [row["scenario"] for row in rows], rotation=30, ha="right")
        plt.ylabel("Final Accuracy Delta vs Full")
        plt.title("Ablation Accuracy Impact")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, "ablation_accuracy_delta.png"))
        plt.close()


def run_ablation_suite(args, output_dir):
    scenarios = _parse_csv_list(args.ablation_scenarios, ABLATION_SCENARIOS, "ablation scenarios")
    if "full" not in scenarios:
        scenarios = ["full"] + scenarios
    method_name = args.ablation_method
    privacy_config = make_privacy_config(args)
    print(f"Running ablation suite: method={method_name}, scenarios={scenarios}")
    print(f"Results will be saved to: {output_dir}")
    _print_privacy_config(privacy_config)

    train_val_data, X_test, y_test, classes, failure_plan = _load_suite_data(args, privacy_config)
    write_data_artifacts(output_dir, args, train_val_data, X_test, y_test, classes, failure_plan)
    scenario_results = {}
    summary_rows = []
    final_rows = []
    full_final_accuracy = None

    for scenario in scenarios:
        scenario_args = _ablation_args(args, scenario)
        scenario_output_dir = os.path.join(output_dir, scenario)
        result = _run_single_method(
            method_name,
            scenario_args,
            train_val_data,
            X_test,
            y_test,
            classes,
            failure_plan,
            scenario_output_dir,
            privacy_config,
            label_suffix=f" ({scenario})",
            config_overrides=_ablation_config_overrides(scenario),
            ablation_scenario=scenario,
        )
        scenario_results[scenario] = result
        if scenario == "full":
            full_final_accuracy = result["accuracies"][-1] if result.get("accuracies") else np.nan
        for idx, round_num in enumerate(result["rounds"]):
            summary_rows.append(
                {
                    "method": method_name,
                    "scenario": scenario,
                    "round": round_num,
                    "accuracy": result["accuracies"][idx],
                    "balanced_accuracy": result["balanced_accuracies"][idx],
                    "f1_score": result["f1_scores"][idx],
                    "auc_roc": result["aucs"][idx],
                    "dp_noise_scale": result["dp_noise_scales"][idx],
                    "zkip_failures": result["zkip_failures"][idx],
                    "ebcd_alert": result["ebcd_alerts"][idx],
                    "tcm_state_count": result["tcm_counts"][idx],
                    "total_regulatory_actions": (
                        result["regulatory_warning_counts"][idx]
                        + result["regulatory_downweight_counts"][idx]
                        + result["regulatory_quarantine_counts"][idx]
                    ),
                    "client_accuracy_gap": result["client_accuracy_gaps"][idx],
                    "avg_contribution_score": result["avg_contribution_scores"][idx],
                }
            )

    if full_final_accuracy is None:
        full_final_accuracy = np.nan
    for scenario, result in scenario_results.items():
        row = _final_metric_row(scenario, result)
        row["method"] = method_name
        row["scenario"] = scenario
        row["disabled_component"] = "none" if scenario == "full" else scenario.replace("no_", "")
        final_accuracy = row.get("final_accuracy", np.nan)
        row["final_accuracy_delta_vs_full"] = (
            final_accuracy - full_final_accuracy
            if np.isfinite(_metric_value(final_accuracy)) and np.isfinite(_metric_value(full_final_accuracy))
            else np.nan
        )
        final_rows.append(row)

    _write_csv(os.path.join(output_dir, "ablation_summary.csv"), summary_rows)
    _write_csv(os.path.join(output_dir, "ablation_final_metrics.csv"), final_rows)
    _plot_ablation_summary(summary_rows, final_rows, output_dir)
    write_artifact_manifest(output_dir)
    print(f"Ablation suite artifacts saved to: {output_dir}")


def run_synthetic_fairness_suite(args, output_dir):
    datasets = _parse_csv_list(args.fairness_datasets, SYNTHETIC_FAIRNESS_DATASETS, "synthetic fairness datasets")
    methods = _parse_csv_list(args.fairness_methods, FAIRNESS_METHODS, "synthetic fairness methods")
    privacy_config = make_privacy_config(args)
    print(f"Running synthetic sensitive-attribute fairness suite: datasets={datasets}, methods={methods}")
    print(f"Results will be saved to: {output_dir}")
    _print_privacy_config(privacy_config)

    method_results = {}
    metadata_rows_all = []
    group_rows_all = []
    federated_rows_all = []

    for dataset_name in datasets:
        data_args = copy.copy(args)
        data_args.dataset = dataset_name
        try:
            data_args, train_val_data, X_test, y_test, classes, failure_plan, metadata_rows, capabilities = (
                _load_synthetic_fairness_data(data_args, privacy_config, dataset_name)
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Synthetic fairness dataset '{dataset_name}' is not ready. "
                f"For FEMNIST/CIFAR10/CIFAR100, generate data/{dataset_name}/all_data first with the dataset generate_data.py script. "
                f"Original error: {exc}"
            ) from exc

        write_data_artifacts(
            output_dir,
            data_args,
            train_val_data,
            X_test,
            y_test,
            classes,
            failure_plan,
            dataset_name=dataset_name,
            metadata_rows=metadata_rows,
        )

        metadata_rows_all.extend(metadata_rows)
        evaluator = GroupFairnessEvaluator(metadata_rows)
        for method_name in methods:
            method_args = copy.copy(data_args)
            method_args.enable_fairness_evaluation = True
            method_args.enable_regulatory_intervention = True
            method_output_dir = os.path.join(output_dir, dataset_name, method_name)
            result = _run_single_method(
                method_name,
                method_args,
                train_val_data,
                X_test,
                y_test,
                classes,
                failure_plan,
                method_output_dir,
                privacy_config,
                label_suffix=f" ({dataset_name} synthetic fairness)",
                synthetic_metadata=metadata_rows,
                capability_overrides=capabilities,
            )
            result_key = f"{dataset_name}_{method_name}"
            group_rows = evaluator.group_rows(dataset_name, method_name, result)
            federated_rows = evaluator.federated_rows(dataset_name, method_name, result)
            _apply_synthetic_final_metrics(result, group_rows, federated_rows)
            method_results[result_key] = result
            group_rows_all.extend(group_rows)
            federated_rows_all.extend(federated_rows)

    _write_csv(
        os.path.join(output_dir, "synthetic_sensitive_clients.csv"),
        [
            {
                "dataset": row["dataset"],
                "client_id": row["client_id"],
                "gender_group": row["gender_group"],
                "age_group": row["age_group"],
                "region_group": row["region_group"],
                "compute_level": row["compute_level"],
                "data_quality": row["data_quality"],
                "sample_count": row["sample_count"],
                "class_coverage": row["class_coverage"],
            }
            for row in metadata_rows_all
        ],
    )
    _write_csv(os.path.join(output_dir, "synthetic_group_fairness_summary.csv"), group_rows_all)
    _write_csv(os.path.join(output_dir, "federated_group_fairness_summary.csv"), federated_rows_all)
    _plot_synthetic_summary(group_rows_all, federated_rows_all, output_dir)
    _save_suite_summary(output_dir, method_results)
    print(f"Synthetic sensitive-attribute fairness suite artifacts saved to: {output_dir}")


def run_participation_suite(args, output_dir):
    policies = _parse_csv_list(args.participation_policies, PARTICIPATION_POLICIES, "participation policies")
    privacy_config = make_privacy_config(args)
    print(f"Running participation suite: {policies}")
    print(f"Results will be saved to: {output_dir}")
    _print_privacy_config(privacy_config)
    train_val_data, X_test, y_test, classes, failure_plan = _load_suite_data(args, privacy_config)
    write_data_artifacts(output_dir, args, train_val_data, X_test, y_test, classes, failure_plan)

    policy_results = {}
    for policy in policies:
        policy_output_dir = os.path.join(output_dir, f"participation_{policy}")
        policy_results[policy] = _run_single_method(
            "grail_fl",
            args,
            train_val_data,
            X_test,
            y_test,
            classes,
            failure_plan,
            policy_output_dir,
            privacy_config,
            participation_policy=policy,
            participation_rate=args.participation_rate,
            label_suffix=f" ({policy})",
        )

    summary_rows = []
    final_rows = []
    for policy, result in policy_results.items():
        final_row = _final_metric_row(policy, result)
        final_row["policy"] = policy
        final_rows.append(final_row)
        for idx, round_num in enumerate(result["rounds"]):
            summary_rows.append(
                {
                    "policy": policy,
                    "round": round_num,
                    "accuracy": result["accuracies"][idx],
                    "balanced_accuracy": result["balanced_accuracies"][idx],
                    "f1_score": result["f1_scores"][idx],
                    "selected_client_count": result["selected_client_counts"][idx],
                    "agg_client_count": result["agg_client_counts"][idx],
                    "dp_noise_scale": result["dp_noise_scales"][idx],
                    "round_duration": result["round_durations"][idx],
                }
            )
    _write_csv(os.path.join(output_dir, "participation_summary.csv"), summary_rows)
    _write_csv(os.path.join(output_dir, "participation_final_metrics.csv"), final_rows)
    _plot_group_metric(summary_rows, "policy", "accuracy", os.path.join(output_dir, "participation_accuracy.png"), "Accuracy", "Participation Policy Accuracy")
    _plot_group_metric(summary_rows, "policy", "selected_client_count", os.path.join(output_dir, "participation_client_count.png"), "Selected Clients", "Participation Policy Client Count")
    _plot_group_metric(summary_rows, "policy", "f1_score", os.path.join(output_dir, "participation_comparison.png"), "Macro-F1", "Participation Policy Macro-F1")
    write_artifact_manifest(output_dir)
    print(f"Participation suite artifacts saved to: {output_dir}")


def _parse_float_list(value, name):
    try:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {value}") from exc


def run_privacy_sensitivity_suite(args, output_dir):
    budgets = _parse_float_list(args.privacy_budgets, "privacy budgets")
    methods = _parse_csv_list(args.privacy_sensitivity_methods, PRIVACY_SENSITIVITY_METHODS, "privacy sensitivity methods")
    print(f"Running privacy sensitivity suite: budgets={budgets}, methods={methods}")
    print(f"Results will be saved to: {output_dir}")

    summary_rows = []
    final_rows = []
    for budget in budgets:
        original_budget = args.total_privacy_budget
        original_per_client_budget = args.epsilon_per_client_total
        args.total_privacy_budget = budget
        args.epsilon_per_client_total = budget
        privacy_config = make_privacy_config(args)
        train_val_data, X_test, y_test, classes, failure_plan = _load_suite_data(args, privacy_config)
        write_data_artifacts(output_dir, args, train_val_data, X_test, y_test, classes, failure_plan)
        budget_results = {}
        for method_name in methods:
            method_output_dir = os.path.join(output_dir, f"budget_{budget:g}", method_name)
            result = _run_single_method(
                method_name,
                args,
                train_val_data,
                X_test,
                y_test,
                classes,
                failure_plan,
                method_output_dir,
                privacy_config,
            )
            budget_results[method_name] = result
            final_row = _final_metric_row(method_name, result)
            final_row["budget"] = budget
            final_row["method"] = method_name
            final_rows.append(final_row)
            for idx, round_num in enumerate(result["rounds"]):
                summary_rows.append(
                    {
                        "budget": budget,
                        "method": method_name,
                        "round": round_num,
                        "accuracy": result["accuracies"][idx],
                        "balanced_accuracy": result["balanced_accuracies"][idx],
                        "f1_score": result["f1_scores"][idx],
                        "dp_noise_scale": result["dp_noise_scales"][idx],
                        "agg_client_count": result["agg_client_counts"][idx],
                        "round_duration": result["round_durations"][idx],
                    }
                )
        _save_suite_summary(os.path.join(output_dir, f"budget_{budget:g}"), budget_results)
        args.total_privacy_budget = original_budget
        args.epsilon_per_client_total = original_per_client_budget

    _write_csv(os.path.join(output_dir, "privacy_sensitivity_summary.csv"), summary_rows)
    _write_csv(os.path.join(output_dir, "privacy_sensitivity_final_metrics.csv"), final_rows)
    _plot_final_metric_by_method(final_rows, "final_accuracy", os.path.join(output_dir, "privacy_budget_accuracy.png"), "Final Accuracy", "Accuracy vs Privacy Budget")
    _plot_final_metric_by_method(final_rows, "avg_dp_noise_scale", os.path.join(output_dir, "privacy_budget_noise.png"), "Avg DP Noise Scale", "Noise vs Privacy Budget")

    for method_name in methods:
        rows = [row for row in final_rows if row["method"] == method_name]
        _plot_final_metric(rows, "budget", "final_accuracy", os.path.join(output_dir, f"privacy_budget_accuracy_{method_name}.png"), "Total Privacy Budget", "Final Accuracy", f"{method_name} Accuracy vs Privacy Budget")
        _plot_final_metric(rows, "budget", "avg_dp_noise_scale", os.path.join(output_dir, f"privacy_budget_noise_{method_name}.png"), "Total Privacy Budget", "Avg DP Noise Scale", f"{method_name} Noise vs Privacy Budget")

    plt.figure(figsize=charts.FIGSIZE_RELATION)
    for method_name in methods:
        rows = sorted([row for row in final_rows if row["method"] == method_name], key=lambda r: r["budget"])
        plt.plot([row["avg_dp_noise_scale"] for row in rows], [row["final_accuracy"] for row in rows], marker="o", label=method_name)
    plt.xlabel("Avg DP Noise Scale")
    plt.ylabel("Final Accuracy")
    plt.title("Privacy-Utility Tradeoff")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    charts.save_figure(os.path.join(output_dir, "privacy_budget_tradeoff.png"))
    plt.close()
    write_artifact_manifest(output_dir)
    print(f"Privacy sensitivity suite artifacts saved to: {output_dir}")

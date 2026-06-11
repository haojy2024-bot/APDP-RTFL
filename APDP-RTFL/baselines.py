import csv
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew, entropy
from sklearn.model_selection import train_test_split

from data_utils import load_experiment_data, split_data_for_clients
from fl_client import FLClient
from fl_server import FLServer
import charting as charts


TOTAL_PRIVACY_BUDGET = 5.0
MIN_EPSILON = 0.1
MAX_EPSILON = 2.0
DP_EPSILON = 1.0
DP_DELTA = 1e-5
DP_L2_NORM_CLIP = 1.0
BASE_LEARNING_RATE = 0.01
EARLYSTOP_PATIENCE = 3
BASELINE_METHODS = ("fedavg", "fedprox", "ldp_fl", "global_dp", "dp_rtfl", "apdp_rtfl")
PARTICIPATION_POLICIES = ("all", "random", "apdp_score")
PRIVACY_SENSITIVITY_METHODS = ("ldp_fl", "global_dp", "dp_rtfl", "apdp_rtfl")


class PrivacyRuntimeConfig:
    def __init__(self, total_budget=TOTAL_PRIVACY_BUDGET, min_epsilon=MIN_EPSILON,
                 max_epsilon=MAX_EPSILON, dp_epsilon=DP_EPSILON, dp_delta=DP_DELTA,
                 dp_l2_norm_clip=DP_L2_NORM_CLIP, failure_prob=0.15,
                 apdp_warmup_rounds=20, adaptive_increase_factor=1.10,
                 adaptive_decrease_factor=0.90, disable_compute_epoch_scaling=False):
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


def make_privacy_config(args):
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
    )


METHOD_CONFIGS = {
    "fedavg": {
        "label": "FedAvg",
        "dp_scope": "none",
        "use_dp": False,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": False,
        "use_ebcd": False,
        "use_tcm": False,
        "fedprox": False,
    },
    "fedprox": {
        "label": "FedProx",
        "dp_scope": "none",
        "use_dp": False,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": False,
        "use_ebcd": False,
        "use_tcm": False,
        "fedprox": True,
    },
    "ldp_fl": {
        "label": "LDP-FL",
        "dp_scope": "client",
        "use_dp": True,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": False,
        "use_ebcd": False,
        "use_tcm": False,
        "fedprox": False,
    },
    "global_dp": {
        "label": "Global-DP",
        "dp_scope": "server",
        "use_dp": False,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": False,
        "use_ebcd": False,
        "use_tcm": False,
        "fedprox": False,
    },
    "dp_rtfl": {
        "label": "DP-RTFL",
        "dp_scope": "client",
        "use_dp": True,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": True,
        "use_ebcd": True,
        "use_tcm": True,
        "fedprox": False,
    },
    "apdp_rtfl": {
        "label": "APDP-RTFL",
        "dp_scope": "client",
        "use_dp": True,
        "dynamic_privacy": True,
        "compute_adapter": True,
        "use_zkip": True,
        "use_ebcd": True,
        "use_tcm": True,
        "fedprox": False,
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
    if not active_indices:
        return {}
    if not dynamic:
        epsilon = privacy_config.total_budget / len(active_indices)
        return {idx: epsilon for idx in active_indices}

    total_data = sum(len(clients[i].y_train) for i in active_indices)
    if total_data == 0:
        epsilon = privacy_config.total_budget / len(active_indices)
        return {idx: epsilon for idx in active_indices}

    compute_factors = {i: 1.0 / (capabilities.get(i, 1.0) + 0.01) for i in active_indices}
    total_compute = sum(compute_factors.values())
    weights = {}
    total_weight = 0.0
    for i in active_indices:
        data_weight = len(clients[i].y_train) / total_data
        quality = _compute_data_quality_score(clients[i].y_train)
        compute_weight = compute_factors[i] / total_compute if total_compute > 0 else 0.0
        weight = 0.50 * data_weight + 0.25 * quality + 0.25 * compute_weight
        weights[i] = weight
        total_weight += weight
    if total_weight == 0:
        epsilon = privacy_config.total_budget / len(active_indices)
        return {idx: epsilon for idx in active_indices}
    return {idx: (weights[idx] / total_weight) * privacy_config.total_budget for idx in active_indices}


def _adaptive_adjust(clients, previous_allocations, round_num, privacy_config):
    if round_num <= privacy_config.apdp_warmup_rounds:
        return previous_allocations.copy()
    scores = {}
    for i, epsilon in previous_allocations.items():
        data_size = len(clients[i].y_train) if hasattr(clients[i], "y_train") else 0
        data_quality = _compute_data_quality_score(clients[i].y_train) if hasattr(clients[i], "y_train") else 0.5
        scores[i] = data_size * data_quality
    global_avg = np.mean(list(scores.values())) if scores else 0.0
    adjusted = previous_allocations.copy()
    for i, epsilon in previous_allocations.items():
        if scores.get(i, 0.0) > global_avg * 1.05:
            adjusted[i] = min(privacy_config.max_epsilon, epsilon * privacy_config.adaptive_increase_factor)
        else:
            adjusted[i] = max(privacy_config.min_epsilon, epsilon * privacy_config.adaptive_decrease_factor)
    total = sum(adjusted.values())
    if total > 0:
        scale = privacy_config.total_budget / total
        adjusted = {
            i: max(privacy_config.min_epsilon, min(privacy_config.max_epsilon, eps * scale))
            for i, eps in adjusted.items()
        }
    return adjusted


def _split_train_val(client_datasets, classes, seed):
    train_val = []
    for i, (X_c, y_c) in enumerate(client_datasets):
        if X_c.shape[0] > 5 and len(np.unique(y_c)) >= 2:
            counts = np.bincount(y_c.astype(int), minlength=len(classes))
            stratify = y_c if np.min(counts[counts > 0]) >= 2 else None
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
            )
        )
    return clients


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


def _aggregate_deltas(base_params, deltas_with_sizes, verify_zkip, zkip, privacy_config=None, apply_server_dp=False):
    valid = []
    total_weight = 0
    aggregated_from = []
    zkip_failures = 0
    for delta, proof, client_id, data_size in deltas_with_sizes:
        if delta is None:
            continue
        if verify_zkip and not zkip.verify_proof(delta, proof):
            zkip_failures += 1
            continue
        valid.append((delta, data_size))
        total_weight += data_size
        aggregated_from.append(client_id)
    if not valid or total_weight == 0:
        return base_params, False, aggregated_from, zkip_failures, 0.0

    aggregated_delta = {k: np.zeros_like(v) for k, v in valid[0][0].items()}
    for delta, data_size in valid:
        weight = data_size / total_weight
        for key in aggregated_delta:
            aggregated_delta[key] += delta[key] * weight
    next_params = {k: np.copy(v) for k, v in base_params.items()}
    server_noise_scale = 0.0
    if apply_server_dp and privacy_config is not None:
        aggregated_delta, server_noise_scale = _apply_server_dp_to_delta(aggregated_delta, privacy_config)
    for key in aggregated_delta:
        next_params[key] += aggregated_delta[key]
    return next_params, True, aggregated_from, zkip_failures, server_noise_scale


def _make_eval_server(method_name, client_ids, num_features, classes, params):
    server = FLServer(f"{method_name}_server", client_ids, num_features, classes=classes)
    server.global_model_parameters = {k: np.copy(v) for k, v in params.items()}
    return server


def _parse_csv_list(value, allowed, name):
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in items if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported {name}: {invalid}. Supported: {allowed}")
    return items


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

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(result), f, indent=2)

    np.save(os.path.join(output_dir, "per_client_update_norms.npy"), np.array(result["per_client_update_norms"], dtype=object))
    np.save(os.path.join(output_dir, "per_client_ebcd_stats.npy"), np.array(result["per_client_ebcd_stats"], dtype=object))
    np.save(os.path.join(output_dir, "per_client_zkip_status.npy"), np.array(result["per_client_zkip_status"], dtype=object))
    np.save(os.path.join(output_dir, "per_client_epsilon.npy"), np.array(result["per_client_epsilon"], dtype=object))

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
):
    config = METHOD_CONFIGS[method_name]
    num_features = X_test.shape[1]
    clients = _init_clients(train_val_data, num_features, classes, args.seed, privacy_config)
    client_ids = [client.client_id for client in clients]
    capabilities = _assign_capabilities(len(clients))
    for i, client in enumerate(clients):
        client.compute_capability = capabilities.get(i, 1.0)

    server = FLServer(f"{method_name}_server", client_ids, num_features, classes=classes)
    if config["use_ebcd"]:
        initial_params = [client.model_parameters() for client in clients if client.X_train.shape[0] > 0]
        if initial_params:
            server.ebcd.establish_baseline(initial_params)

    active_indices = list(range(len(clients)))
    current_allocations = _allocate_budget(clients, active_indices, capabilities, config["dynamic_privacy"], privacy_config)
    contribution_history = []

    result = {
        "method": method_name,
        "label": f"{config['label']}{label_suffix}",
        "participation_policy": participation_policy,
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
    }

    print(f"\n=== Running {config['label']} ({method_name}) ===")
    rng = np.random.default_rng(args.seed + 5000)
    contribution_scores = {idx: 0.0 for idx in range(len(clients))}
    for round_num in range(1, args.num_rounds + 1):
        start_time = time.time()
        if config["dynamic_privacy"] and round_num > 1:
            current_allocations = _adaptive_adjust(clients, current_allocations, round_num, privacy_config)

        global_params = {k: np.copy(v) for k, v in server.global_model_parameters.items()}
        client_updates = []
        round_update_norms = []
        round_ebcd_stats = []
        round_zkip_status = []
        round_epsilons = []
        round_noise_scales = []
        round_delta_norm = 0.0
        available_indices = [idx for idx in range(len(clients)) if not failure_plan[round_num - 1][idx]]
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

        for idx, client in enumerate(clients):
            if failure_plan[round_num - 1][idx] or idx not in selected_indices:
                round_update_norms.append(None)
                round_ebcd_stats.append((None, None, None))
                round_zkip_status.append(None)
                round_epsilons.append(None)
                continue

            base_epsilon = current_allocations.get(idx, privacy_config.dp_epsilon)
            effective_epsilon = _adjust_epsilon_for_compute(
                base_epsilon, idx, capabilities, config["compute_adapter"], privacy_config
            )
            client.dp_epsilon = max(privacy_config.min_epsilon, min(privacy_config.max_epsilon, effective_epsilon))
            round_epsilons.append(client.dp_epsilon)
            effective_epochs = _effective_epochs(
                idx, args.client_epochs, capabilities, config["compute_adapter"], privacy_config
            )
            client.set_global_model_parameters(global_params)
            delta, proof = client.train(
                epochs=effective_epochs,
                use_dp=config["dp_scope"] == "client",
                fedprox_mu=args.fedprox_mu if config["fedprox"] else 0.0,
                global_params=global_params,
            )
            if delta is None:
                round_update_norms.append(None)
                round_ebcd_stats.append((None, None, None))
                round_zkip_status.append(False)
                continue

            update_norm = np.sqrt(sum(np.linalg.norm(v.flatten()) ** 2 for v in delta.values()))
            round_update_norms.append(update_norm)
            round_delta_norm += update_norm
            if "coef_" in delta and hasattr(delta["coef_"], "flatten"):
                flat = delta["coef_"].flatten()
                round_ebcd_stats.append((np.var(flat), kurtosis(flat, fisher=True), skew(flat)))
            else:
                round_ebcd_stats.append((None, None, None))
            zkip_ok = server.zkip.verify_proof(delta, proof) if config["use_zkip"] else True
            round_zkip_status.append(zkip_ok)
            client_updates.append((delta, proof, client.client_id, len(client.y_train)))
            if config["dp_scope"] == "client" and client.dp_epsilon > 0:
                noise_stddev = (client.dp_l2_norm_clip * np.sqrt(2 * np.log(1.25 / client.dp_delta))) / client.dp_epsilon
            else:
                noise_stddev = 0.0
            round_noise_scales.append(noise_stddev)

        server.global_model_parameters, aggregation_success, aggregated_from, zkip_failures, server_noise_scale = _aggregate_deltas(
            server.global_model_parameters,
            client_updates,
            config["use_zkip"],
            server.zkip,
            privacy_config=privacy_config,
            apply_server_dp=config["dp_scope"] == "server",
        )
        ebcd_alert = 1 if config["use_ebcd"] and server.ebcd.check_for_corruption(server.global_model_parameters) else 0
        if config["use_tcm"]:
            state_details = {
                "method": method_name,
                "aggregation_successful": aggregation_success,
                "aggregated_from_clients_count": len(aggregated_from),
            }
            server.tcm.record_state(
                round_num,
                server.global_model_parameters,
                state_details,
                {cid: "OK" for cid in aggregated_from},
            )

        eval_server = _make_eval_server(method_name, client_ids, num_features, classes, server.global_model_parameters)
        metrics = eval_server.evaluate_global_model(X_test, y_test, round_num)
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
    invalid = [m for m in methods if m not in BASELINE_METHODS]
    if invalid:
        raise ValueError(f"Unsupported baseline methods: {invalid}. Supported: {BASELINE_METHODS}")
    return methods


def _final_metric_row(method_name, result):
    return {
        "method": method_name,
        "label": result["label"],
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
    }


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


def _write_csv(path, rows):
    fieldnames = list(rows[0].keys()) if rows else ["name"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        f"total_budget={privacy_config.total_budget}, "
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

    train_val_data, X_test, y_test, classes, failure_plan = _load_suite_data(args, privacy_config)

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


def run_participation_suite(args, output_dir):
    policies = _parse_csv_list(args.participation_policies, PARTICIPATION_POLICIES, "participation policies")
    privacy_config = make_privacy_config(args)
    print(f"Running participation suite: {policies}")
    print(f"Results will be saved to: {output_dir}")
    _print_privacy_config(privacy_config)
    train_val_data, X_test, y_test, classes, failure_plan = _load_suite_data(args, privacy_config)

    policy_results = {}
    for policy in policies:
        policy_output_dir = os.path.join(output_dir, f"participation_{policy}")
        policy_results[policy] = _run_single_method(
            "apdp_rtfl",
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
        args.total_privacy_budget = budget
        privacy_config = make_privacy_config(args)
        train_val_data, X_test, y_test, classes, failure_plan = _load_suite_data(args, privacy_config)
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
    print(f"Privacy sensitivity suite artifacts saved to: {output_dir}")

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
BASELINE_METHODS = ("fedavg", "fedprox", "ldp_fl", "dp_rtfl", "apdp_rtfl")


class PrivacyRuntimeConfig:
    def __init__(self, total_budget=TOTAL_PRIVACY_BUDGET, min_epsilon=MIN_EPSILON,
                 max_epsilon=MAX_EPSILON, dp_epsilon=DP_EPSILON, dp_delta=DP_DELTA,
                 dp_l2_norm_clip=DP_L2_NORM_CLIP, failure_prob=0.15):
        self.total_budget = total_budget
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_l2_norm_clip = dp_l2_norm_clip
        self.failure_prob = failure_prob


def make_privacy_config(args):
    return PrivacyRuntimeConfig(
        total_budget=args.total_privacy_budget,
        min_epsilon=args.min_epsilon,
        max_epsilon=args.max_epsilon,
        dp_epsilon=args.dp_epsilon,
        dp_delta=args.dp_delta,
        dp_l2_norm_clip=args.dp_l2_norm_clip,
        failure_prob=args.failure_prob,
    )


METHOD_CONFIGS = {
    "fedavg": {
        "label": "FedAvg",
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
        "use_dp": True,
        "dynamic_privacy": False,
        "compute_adapter": False,
        "use_zkip": False,
        "use_ebcd": False,
        "use_tcm": False,
        "fedprox": False,
    },
    "dp_rtfl": {
        "label": "DP-RTFL",
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


def _effective_epochs(client_idx, base_epochs, capabilities, enabled):
    if not enabled:
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


def _adaptive_adjust(clients, previous_allocations, contribution_history, privacy_config):
    if not contribution_history:
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
            adjusted[i] = min(privacy_config.max_epsilon, epsilon * 1.25)
        else:
            adjusted[i] = max(privacy_config.min_epsilon, epsilon * 0.75)
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


def _aggregate_deltas(base_params, deltas_with_sizes, verify_zkip, zkip):
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
        return base_params, False, aggregated_from, zkip_failures

    aggregated_delta = {k: np.zeros_like(v) for k, v in valid[0][0].items()}
    for delta, data_size in valid:
        weight = data_size / total_weight
        for key in aggregated_delta:
            aggregated_delta[key] += delta[key] * weight
    next_params = {k: np.copy(v) for k, v in base_params.items()}
    for key in aggregated_delta:
        next_params[key] += aggregated_delta[key]
    return next_params, True, aggregated_from, zkip_failures


def _make_eval_server(method_name, client_ids, num_features, classes, params):
    server = FLServer(f"{method_name}_server", client_ids, num_features, classes=classes)
    server.global_model_parameters = {k: np.copy(v) for k, v in params.items()}
    return server


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
    plt.savefig(os.path.join(output_dir, "global_metrics.png"))
    plt.close()
    charts.plot_dp_noise_scale(result["rounds"], result["dp_noise_scales"])
    plt.savefig(os.path.join(output_dir, "dp_noise_scale.png"))
    plt.close()
    charts.plot_agg_client_counts(result["rounds"], result["agg_client_counts"])
    plt.savefig(os.path.join(output_dir, "agg_client_counts.png"))
    plt.close()
    charts.plot_zkip_failures(result["rounds"], result["zkip_failures"])
    plt.savefig(os.path.join(output_dir, "zkip_failures.png"))
    plt.close()
    charts.plot_delta_norm(result["rounds"], result["delta_norms"])
    plt.savefig(os.path.join(output_dir, "delta_norm.png"))
    plt.close()
    charts.plot_ebcd_alerts(result["rounds"], result["ebcd_alerts"])
    plt.savefig(os.path.join(output_dir, "ebcd_alerts.png"))
    plt.close()
    charts.plot_tcm_state_count(result["rounds"], result["tcm_counts"])
    plt.savefig(os.path.join(output_dir, "tcm_state_count.png"))
    plt.close()
    charts.plot_per_client_update_norms(result["rounds"], result["per_client_update_norms"], client_ids)
    plt.savefig(os.path.join(output_dir, "per_client_update_norms.png"))
    plt.close()


def _run_single_method(method_name, args, train_val_data, X_test, y_test, classes, failure_plan, output_dir, privacy_config):
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
        "label": config["label"],
        "rounds": [],
        "accuracies": [],
        "balanced_accuracies": [],
        "f1_scores": [],
        "precisions": [],
        "recalls": [],
        "aucs": [],
        "round_durations": [],
        "agg_client_counts": [],
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
    for round_num in range(1, args.num_rounds + 1):
        start_time = time.time()
        if config["dynamic_privacy"] and round_num > 1:
            current_allocations = _adaptive_adjust(clients, current_allocations, contribution_history, privacy_config)

        global_params = {k: np.copy(v) for k, v in server.global_model_parameters.items()}
        client_updates = []
        round_update_norms = []
        round_ebcd_stats = []
        round_zkip_status = []
        round_epsilons = []
        round_noise_scales = []
        round_delta_norm = 0.0

        for idx, client in enumerate(clients):
            if failure_plan[round_num - 1][idx]:
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
                idx, args.client_epochs, capabilities, config["compute_adapter"]
            )
            client.set_global_model_parameters(global_params)
            delta, proof = client.train(
                epochs=effective_epochs,
                use_dp=config["use_dp"],
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
            if config["use_dp"] and client.dp_epsilon > 0:
                noise_stddev = (client.dp_l2_norm_clip * np.sqrt(2 * np.log(1.25 / client.dp_delta))) / client.dp_epsilon
            else:
                noise_stddev = 0.0
            round_noise_scales.append(noise_stddev)

        server.global_model_parameters, aggregation_success, aggregated_from, zkip_failures = _aggregate_deltas(
            server.global_model_parameters, client_updates, config["use_zkip"], server.zkip
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
        result["dp_noise_scales"].append(float(np.mean(round_noise_scales)) if round_noise_scales else 0.0)
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

    plt.figure(figsize=(12, 8))
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
    plt.savefig(os.path.join(output_dir, "baseline_comparison.png"))
    plt.close()


def run_baseline_suite(args, output_dir):
    methods = _parse_methods(args.methods)
    privacy_config = make_privacy_config(args)
    print(f"Running baseline suite: {methods}")
    print(f"Results will be saved to: {output_dir}")
    print(
        "Privacy config: "
        f"total_budget={privacy_config.total_budget}, "
        f"min_epsilon={privacy_config.min_epsilon}, "
        f"max_epsilon={privacy_config.max_epsilon}, "
        f"dp_l2_norm_clip={privacy_config.dp_l2_norm_clip}, "
        f"failure_prob={privacy_config.failure_prob}"
    )

    X_train_full, y_train_full, X_test, y_test, _, classes, presplit_client_data = load_experiment_data(
        dataset_name=args.dataset,
        data_root=args.data_root,
        random_state=args.seed,
        max_samples=args.max_samples,
        emnist_split=args.emnist_split,
    )
    if X_train_full is None:
        print("Failed to load data. Exiting.")
        return
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

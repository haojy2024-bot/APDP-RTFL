import os
import time

import numpy as np
import torch
from scipy.stats import kurtosis, skew
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from baselines import (
    BASELINE_METHODS,
    BASE_LEARNING_RATE,
    METHOD_CONFIGS,
    _adaptive_adjust,
    _adjust_epsilon_for_compute,
    _aggregate_deltas,
    _allocate_budget,
    _assign_capabilities,
    _effective_epochs,
    _metric_value,
    _parse_methods,
    _save_method_artifacts,
    _save_suite_summary,
    _split_train_val,
    make_privacy_config,
)
from data_utils import load_experiment_data, split_data_for_clients
from fl_server import FLServer
from experiment_artifacts import write_data_artifacts, write_artifact_manifest


class TorchLinearClient:
    def __init__(
        self,
        client_id,
        X_train,
        y_train,
        num_features,
        classes,
        device,
        learning_rate=BASE_LEARNING_RATE,
        batch_size=256,
        random_state=0,
        privacy_config=None,
    ):
        self.client_id = client_id
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=int)
        self.num_features = num_features
        self.classes = np.asarray(classes, dtype=int)
        self.n_classes = len(self.classes)
        self.class_to_index = {int(label): idx for idx, label in enumerate(self.classes)}
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.dp_epsilon = privacy_config.dp_epsilon
        self.dp_delta = privacy_config.dp_delta
        self.dp_l2_norm_clip = privacy_config.dp_l2_norm_clip
        self.model = torch.nn.Linear(num_features, self.n_classes).to(device)
        torch.manual_seed(random_state)
        with torch.no_grad():
            self.model.weight.zero_()
            self.model.bias.zero_()
        from zkip import ZeroKnowledgeIntegrityProofs

        self.zkip = ZeroKnowledgeIntegrityProofs()
        self.last_val_acc_gain = 0.0
        self.last_val_loss_drop = 0.0

    def model_parameters(self):
        with torch.no_grad():
            return {
                "coef_": self.model.weight.detach().cpu().numpy().copy(),
                "intercept_": self.model.bias.detach().cpu().numpy().copy(),
            }

    def set_global_model_parameters(self, global_params):
        with torch.no_grad():
            weight = torch.tensor(global_params["coef_"], dtype=torch.float32, device=self.device)
            bias = torch.tensor(global_params["intercept_"], dtype=torch.float32, device=self.device)
            self.model.weight.copy_(weight)
            self.model.bias.copy_(bias)

    def _encode_y(self, y):
        return np.asarray([self.class_to_index[int(label)] for label in y], dtype=np.int64)

    def _apply_differential_privacy(self, delta_params):
        total_norm = np.sqrt(sum(np.linalg.norm(v.flatten()) ** 2 for v in delta_params.values()))
        clip_factor = min(1.0, self.dp_l2_norm_clip / (total_norm + 1e-6))
        if self.dp_epsilon > 0:
            noise_stddev = (self.dp_l2_norm_clip * np.sqrt(2 * np.log(1.25 / self.dp_delta))) / self.dp_epsilon
        else:
            noise_stddev = 0.0
        noisy = {}
        for key, value in delta_params.items():
            clipped = value * clip_factor
            noisy[key] = clipped + np.random.normal(0, noise_stddev, size=value.shape)
        return noisy

    def train(self, global_params, epochs, use_dp=True, fedprox_mu=0.0):
        if self.X_train.shape[0] == 0:
            return None, None
        self.set_global_model_parameters(global_params)
        global_weight = torch.tensor(global_params["coef_"], dtype=torch.float32, device=self.device)
        global_bias = torch.tensor(global_params["intercept_"], dtype=torch.float32, device=self.device)
        X = torch.tensor(self.X_train, dtype=torch.float32, device=self.device)
        y = torch.tensor(self._encode_y(self.y_train), dtype=torch.long, device=self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        generator = torch.Generator()
        generator.manual_seed(self.random_state)

        for _ in range(epochs):
            permutation = torch.randperm(X.shape[0], generator=generator).to(self.device)
            for start in range(0, X.shape[0], self.batch_size):
                idx = permutation[start:start + self.batch_size]
                logits = self.model(X[idx])
                loss = torch.nn.functional.cross_entropy(logits, y[idx])
                if fedprox_mu > 0:
                    prox = torch.sum((self.model.weight - global_weight) ** 2)
                    prox = prox + torch.sum((self.model.bias - global_bias) ** 2)
                    loss = loss + 0.5 * fedprox_mu * prox
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        current_params = self.model_parameters()
        delta = {key: current_params[key] - global_params[key] for key in current_params}
        if use_dp and self.dp_epsilon > 0:
            delta = self._apply_differential_privacy(delta)
        proof = self.zkip.generate_proof(delta)
        return delta, proof


def _initial_params(num_features, n_classes):
    return {
        "coef_": np.zeros((n_classes, num_features), dtype=np.float32),
        "intercept_": np.zeros(n_classes, dtype=np.float32),
    }


def _softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _evaluate_params(params, X_test, y_test, classes):
    logits = np.asarray(X_test, dtype=np.float32) @ params["coef_"].T + params["intercept_"]
    probas = _softmax(logits)
    pred_indices = np.argmax(probas, axis=1)
    predictions = classes[pred_indices]
    y_classes = np.unique(y_test)
    pred_classes = np.unique(predictions)
    average = "binary" if len(classes) <= 2 else "macro"
    if len(y_classes) < 2 or len(pred_classes) < 2:
        f1 = np.nan
        precision = np.nan
        recall = np.nan
        auc = np.nan
    else:
        f1 = f1_score(y_test, predictions, average=average, zero_division=0)
        precision = precision_score(y_test, predictions, average=average, zero_division=0)
        recall = recall_score(y_test, predictions, average=average, zero_division=0)
        try:
            if len(classes) <= 2:
                auc = roc_auc_score(y_test, probas[:, 1])
            elif len(y_classes) < len(classes):
                auc = np.nan
            else:
                auc = roc_auc_score(y_test, probas, multi_class="ovr", average="macro", labels=classes)
        except ValueError:
            auc = np.nan
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc,
    }


def _resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def _init_torch_clients(train_val_data, num_features, classes, device, args, privacy_config):
    clients = []
    for i, (X_train, y_train, _, _) in enumerate(train_val_data):
        clients.append(
            TorchLinearClient(
                f"client_{i}",
                X_train,
                y_train,
                num_features,
                classes,
                device,
                learning_rate=BASE_LEARNING_RATE,
                batch_size=args.torch_batch_size,
                random_state=args.seed + i,
                privacy_config=privacy_config,
            )
        )
    return clients


def _run_single_torch_method(method_name, args, train_val_data, X_test, y_test, classes, failure_plan, output_dir, device, privacy_config):
    config = METHOD_CONFIGS[method_name]
    num_features = X_test.shape[1]
    classes = np.asarray(classes, dtype=int)
    clients = _init_torch_clients(train_val_data, num_features, classes, device, args, privacy_config)
    client_ids = [client.client_id for client in clients]
    capabilities = _assign_capabilities(len(clients))
    server = FLServer(f"{method_name}_server", client_ids, num_features, classes=classes)
    server.global_model_parameters = _initial_params(num_features, len(classes))
    if config["use_ebcd"]:
        server.ebcd.establish_baseline([client.model_parameters() for client in clients])

    active_indices = list(range(len(clients)))
    current_allocations = _allocate_budget(clients, active_indices, capabilities, config["dynamic_privacy"], privacy_config)
    contribution_history = []
    result = {
        "method": method_name,
        "label": f"{config['label']} (torch)",
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
        "tcm": server.tcm,
    }

    print(f"\n=== Running {config['label']} ({method_name}, torch, device={device}) ===")
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
                idx, args.client_epochs, capabilities, config["compute_adapter"], privacy_config
            )
            delta, proof = client.train(
                global_params=global_params,
                epochs=effective_epochs,
                use_dp=config["dp_scope"] == "client",
                fedprox_mu=args.fedprox_mu if config["fedprox"] else 0.0,
            )
            if delta is None:
                round_update_norms.append(None)
                round_ebcd_stats.append((None, None, None))
                round_zkip_status.append(False)
                continue

            update_norm = np.sqrt(sum(np.linalg.norm(v.flatten()) ** 2 for v in delta.values()))
            round_update_norms.append(update_norm)
            round_delta_norm += update_norm
            flat = delta["coef_"].flatten()
            round_ebcd_stats.append((np.var(flat), float(kurtosis(flat, fisher=True)), float(skew(flat))))
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
            server.tcm.record_state(
                round_num,
                server.global_model_parameters,
                {
                    "method": method_name,
                    "backend": "torch",
                    "device": str(device),
                    "aggregation_successful": aggregation_success,
                    "aggregated_from_clients_count": len(aggregated_from),
                },
                {cid: "OK" for cid in aggregated_from},
            )

        metrics = _evaluate_params(server.global_model_parameters, X_test, y_test, classes)
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
        result["selected_client_counts"].append(len(aggregated_from))
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
        contribution_history.append({idx: round_update_norms[idx] or 0.0 for idx in range(len(clients))})

        print(
            f"{config['label']} round {round_num}/{args.num_rounds}: "
            f"Acc={result['accuracies'][-1]:.3f}, F1={result['f1_scores'][-1]:.3f}, "
            f"BalancedAcc={result['balanced_accuracies'][-1]:.3f}"
        )

    _save_method_artifacts(output_dir, method_name, result, client_ids)
    return result


def run_torch_baseline_suite(args, output_dir):
    device = _resolve_device(args.device)
    methods = _parse_methods(args.methods)
    privacy_config = make_privacy_config(args)
    print(f"Running torch baseline suite on device={device}: {methods}")
    print(f"Results will be saved to: {output_dir}")
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
    write_data_artifacts(output_dir, args, train_val_data, X_test, y_test, classes, failure_plan)
    method_results = {}
    for method_name in methods:
        method_output_dir = os.path.join(output_dir, method_name)
        method_results[method_name] = _run_single_torch_method(
            method_name,
            args,
            train_val_data,
            X_test,
            y_test,
            classes,
            failure_plan,
            method_output_dir,
            device,
            privacy_config,
        )
    _save_suite_summary(output_dir, method_results)
    write_artifact_manifest(output_dir)
    print(f"Torch baseline suite artifacts saved to: {output_dir}")

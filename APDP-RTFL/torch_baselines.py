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
    _local_step_count,
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
from privacy_accounting import RDPAccountant, calibrate_noise_multiplier


class TorchLinearClient:
    def __init__(
        self,
        client_id,
        X_train,
        y_train,
        num_features,
        classes,
        device,
        X_val=None,
        y_val=None,
        learning_rate=BASE_LEARNING_RATE,
        batch_size=256,
        random_state=0,
        privacy_config=None,
        model_type="linear",
        mlp_hidden=(256, 128),
        cnn_channels=(16, 32),
        cnn_fc=128,
    ):
        self.client_id = client_id
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=int)
        self.X_val = None if X_val is None else np.asarray(X_val, dtype=np.float32)
        self.y_val = None if y_val is None else np.asarray(y_val, dtype=int)
        self.num_features = num_features
        self.classes = np.asarray(classes, dtype=int)
        self.n_classes = len(self.classes)
        self.class_to_index = {int(label): idx for idx, label in enumerate(self.classes)}
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.model_type = model_type
        self.mlp_hidden = tuple(int(v) for v in mlp_hidden)
        self.cnn_channels = tuple(int(v) for v in cnn_channels)
        self.cnn_fc = int(cnn_fc)
        self.image_shape = _infer_image_shape(num_features)
        self.dp_epsilon = privacy_config.dp_epsilon
        self.dp_delta = privacy_config.dp_delta
        self.dp_l2_norm_clip = privacy_config.dp_l2_norm_clip
        self.dp_batch_size = int(privacy_config.dp_batch_size)
        self.model = self._build_model().to(device)
        torch.manual_seed(random_state)
        self._initialize_model()
        from zkip import ZeroKnowledgeIntegrityProofs

        self.zkip = ZeroKnowledgeIntegrityProofs()
        self.last_val_acc_gain = 0.0
        self.last_val_loss_drop = 0.0
        self.last_privacy_event = None

    def _build_model(self):
        if self.model_type == "linear":
            return torch.nn.Linear(self.num_features, self.n_classes)
        if self.model_type == "mlp":
            layers = []
            in_features = self.num_features
            for hidden in self.mlp_hidden:
                layers.append(torch.nn.Linear(in_features, hidden))
                layers.append(torch.nn.ReLU())
                in_features = hidden
            layers.append(torch.nn.Linear(in_features, self.n_classes))
            return torch.nn.Sequential(*layers)
        if self.model_type == "cnn":
            channels, height, width = self.image_shape
            if height < 4 or width < 4:
                raise ValueError("--torch-model cnn requires image-like flattened input.")
            if len(self.cnn_channels) != 2:
                raise ValueError("--torch-cnn-channels currently expects exactly two comma-separated values, e.g. 32,64.")
            conv1_out, conv2_out = self.cnn_channels
            return torch.nn.Sequential(
                torch.nn.Unflatten(1, (channels, height, width)),
                torch.nn.Conv2d(channels, conv1_out, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Flatten(),
                torch.nn.Linear(conv2_out * (height // 4) * (width // 4), self.cnn_fc),
                torch.nn.ReLU(),
                torch.nn.Linear(self.cnn_fc, self.n_classes),
            )
        raise ValueError(f"Unsupported torch model type: {self.model_type}")

    def _initialize_model(self):
        torch.manual_seed(self.random_state)
        with torch.no_grad():
            if self.model_type == "linear":
                self.model.weight.zero_()
                self.model.bias.zero_()
                return
            for module in self.model.modules():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    module.bias.zero_()
                elif isinstance(module, torch.nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    module.bias.zero_()

    def model_parameters(self):
        with torch.no_grad():
            if self.model_type == "linear":
                return {
                    "coef_": self.model.weight.detach().cpu().numpy().copy(),
                    "intercept_": self.model.bias.detach().cpu().numpy().copy(),
                }
            return {
                name: value.detach().cpu().numpy().copy()
                for name, value in self.model.state_dict().items()
            }

    def set_global_model_parameters(self, global_params):
        with torch.no_grad():
            if self.model_type == "linear":
                weight = torch.tensor(global_params["coef_"], dtype=torch.float32, device=self.device)
                bias = torch.tensor(global_params["intercept_"], dtype=torch.float32, device=self.device)
                self.model.weight.copy_(weight)
                self.model.bias.copy_(bias)
                return
            state = {
                name: torch.tensor(value, dtype=torch.float32, device=self.device)
                for name, value in global_params.items()
            }
            self.model.load_state_dict(state, strict=True)

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

    def _make_generator(self, offset=0):
        try:
            generator = torch.Generator(device=self.device)
        except TypeError:
            generator = torch.Generator()
        generator.manual_seed(int(self.random_state) + int(offset))
        return generator

    def _train_with_dp_sgd(self, global_params, epochs, fedprox_mu, accountant, noise_multiplier, round_num):
        n_samples = len(self.y_train)
        if n_samples == 0:
            return None, None
        if accountant is None or noise_multiplier is None:
            raise ValueError("DP-SGD requires both an RDP accountant and a noise multiplier.")

        batch_size = min(max(1, int(self.dp_batch_size)), n_samples)
        sample_rate = batch_size / n_samples
        steps = int(epochs) * int(np.ceil(n_samples / batch_size))
        event = accountant.spend(round_num or 0, steps, sample_rate, noise_multiplier)
        self.last_privacy_event = event
        if event.status != "spent":
            return None, None

        self.set_global_model_parameters(global_params)
        X = torch.tensor(self.X_train, dtype=torch.float32, device=self.device)
        y = torch.tensor(self._encode_y(self.y_train), dtype=torch.long, device=self.device)
        generator = self._make_generator(10007 * int(round_num or 1))
        noise_std = float(noise_multiplier) * float(self.dp_l2_norm_clip)

        if self.model_type in {"mlp", "cnn"}:
            return self._train_functional_model_with_dp_sgd(
                global_params, X, y, epochs, fedprox_mu, generator, noise_std
            )

        global_weight = torch.tensor(global_params["coef_"], dtype=torch.float32, device=self.device)
        global_bias = torch.tensor(global_params["intercept_"], dtype=torch.float32, device=self.device)
        weight = self.model.weight
        bias = self.model.bias
        with torch.no_grad():
            for _ in range(int(epochs)):
                for _ in range(int(np.ceil(n_samples / batch_size))):
                    selected = torch.rand(n_samples, device=self.device, generator=generator) < sample_rate
                    if not bool(selected.any().item()):
                        selected[torch.randint(n_samples, (1,), device=self.device, generator=generator)] = True
                    X_batch = X[selected]
                    y_batch = y[selected]
                    logits = torch.nn.functional.linear(X_batch, weight, bias)
                    probabilities = torch.softmax(logits, dim=1)
                    residual = probabilities
                    residual[torch.arange(y_batch.shape[0], device=self.device), y_batch] -= 1.0
                    grad_w_each = residual[:, :, None] * X_batch[:, None, :]
                    grad_b_each = residual
                    norms = torch.sqrt(
                        grad_w_each.square().sum(dim=(1, 2)) + grad_b_each.square().sum(dim=1)
                    )
                    scales = torch.clamp(float(self.dp_l2_norm_clip) / (norms + 1e-12), max=1.0)
                    denominator = max(1, int(y_batch.shape[0]))
                    grad_w = (grad_w_each * scales[:, None, None]).mean(dim=0)
                    grad_b = (grad_b_each * scales[:, None]).mean(dim=0)
                    grad_w = grad_w + torch.randn(grad_w.shape, device=self.device, generator=generator) * (noise_std / denominator)
                    grad_b = grad_b + torch.randn(grad_b.shape, device=self.device, generator=generator) * (noise_std / denominator)
                    if fedprox_mu > 0:
                        grad_w = grad_w + fedprox_mu * (weight - global_weight)
                        grad_b = grad_b + fedprox_mu * (bias - global_bias)
                    weight -= self.learning_rate * grad_w
                    bias -= self.learning_rate * grad_b

        current_params = self.model_parameters()
        delta = {key: current_params[key] - global_params[key] for key in current_params}
        proof = self.zkip.generate_proof(delta)
        return delta, proof

    def _train_functional_model_with_dp_sgd(self, global_params, X, y, epochs, fedprox_mu, generator, noise_std):
        try:
            from torch.func import functional_call, grad, vmap
        except ImportError as exc:
            raise RuntimeError("--torch-model mlp/cnn requires a PyTorch build with torch.func support.") from exc

        batch_size = min(max(1, int(self.dp_batch_size)), len(self.y_train))
        sample_rate = batch_size / max(1, len(self.y_train))
        global_tensors = {
            name: torch.tensor(value, dtype=torch.float32, device=self.device)
            for name, value in global_params.items()
        }

        def loss_one(params, x_one, y_one):
            logits = functional_call(self.model, params, (x_one.unsqueeze(0),))
            return torch.nn.functional.cross_entropy(logits, y_one.unsqueeze(0))

        grad_one = grad(loss_one)
        grad_many = vmap(grad_one, in_dims=(None, 0, 0))

        for _ in range(int(epochs)):
            for _ in range(int(np.ceil(len(self.y_train) / batch_size))):
                selected = torch.rand(len(self.y_train), device=self.device, generator=generator) < sample_rate
                if not bool(selected.any().item()):
                    selected[torch.randint(len(self.y_train), (1,), device=self.device, generator=generator)] = True
                X_batch = X[selected]
                y_batch = y[selected]
                params = {name: param for name, param in self.model.named_parameters()}
                per_sample_grads = grad_many(params, X_batch, y_batch)
                norms = None
                for grad_values in per_sample_grads.values():
                    flat = grad_values.reshape(grad_values.shape[0], -1)
                    term = flat.square().sum(dim=1)
                    norms = term if norms is None else norms + term
                norms = torch.sqrt(norms + 1e-12)
                scales = torch.clamp(float(self.dp_l2_norm_clip) / (norms + 1e-12), max=1.0)
                denominator = max(1, int(y_batch.shape[0]))

                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        grad_values = per_sample_grads[name]
                        view_shape = (grad_values.shape[0],) + (1,) * (grad_values.ndim - 1)
                        clipped_mean = (grad_values * scales.reshape(view_shape)).mean(dim=0)
                        noise = torch.randn(
                            clipped_mean.shape,
                            device=self.device,
                            generator=generator,
                        ) * (noise_std / denominator)
                        update = clipped_mean + noise
                        if fedprox_mu > 0:
                            update = update + fedprox_mu * (param - global_tensors[name])
                        param -= self.learning_rate * update

        current_params = self.model_parameters()
        delta = {key: current_params[key] - global_params[key] for key in current_params}
        proof = self.zkip.generate_proof(delta)
        return delta, proof

    def train(self, global_params, epochs, use_dp=True, fedprox_mu=0.0,
              privacy_accountant=None, noise_multiplier=None, round_num=None):
        if self.X_train.shape[0] == 0:
            return None, None
        self.last_privacy_event = None
        if use_dp and noise_multiplier is not None:
            return self._train_with_dp_sgd(
                global_params,
                epochs,
                fedprox_mu,
                privacy_accountant,
                noise_multiplier,
                round_num,
            )
        self.set_global_model_parameters(global_params)
        global_tensors = {
            name: torch.tensor(value, dtype=torch.float32, device=self.device)
            for name, value in global_params.items()
        }
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
                    prox = None
                    for name, param in self.model.named_parameters():
                        term = torch.sum((param - global_tensors[name]) ** 2)
                        prox = term if prox is None else prox + term
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


def _parse_mlp_hidden(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _parse_cnn_channels(value):
    if isinstance(value, (list, tuple)):
        channels = tuple(int(v) for v in value)
    else:
        channels = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if len(channels) != 2:
        raise ValueError("--torch-cnn-channels expects exactly two comma-separated values, e.g. 32,64.")
    return channels


def _infer_image_shape(num_features):
    if num_features == 784:
        return (1, 28, 28)
    if num_features == 3072:
        return (3, 32, 32)
    side = int(round(float(num_features) ** 0.5))
    if side * side == num_features:
        return (1, side, side)
    raise ValueError(
        f"--torch-model cnn requires flattened square images or CIFAR-like 3072 features; got {num_features}."
    )


def _flatten_params(params):
    arrays = []
    for key in sorted(params):
        value = np.asarray(params[key])
        arrays.append(value.reshape(-1))
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float32)


def _softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _evaluate_params(params, X_test, y_test, classes):
    logits = _predict_logits_from_params(params, X_test)
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


def _predict_logits_from_params(params, X_data):
    if "coef_" in params and "intercept_" in params:
        return np.asarray(X_data, dtype=np.float32) @ params["coef_"].T + params["intercept_"]
    if any(key.startswith("1.") and key.endswith(".weight") for key in params) and any(
        key.startswith("4.") and key.endswith(".weight") for key in params
    ):
        return _predict_logits_from_torch_state(params, X_data)
    activations = np.asarray(X_data, dtype=np.float32)
    layer_indices = sorted(
        int(key.split(".")[0])
        for key in params
        if key.endswith(".weight") and key.split(".")[0].isdigit()
    )
    linear_layers = [idx for idx in layer_indices if f"{idx}.bias" in params]
    for layer_pos, idx in enumerate(linear_layers):
        activations = activations @ np.asarray(params[f"{idx}.weight"], dtype=np.float32).T
        activations = activations + np.asarray(params[f"{idx}.bias"], dtype=np.float32)
        if layer_pos < len(linear_layers) - 1:
            activations = np.maximum(activations, 0.0)
    return activations


def _predict_logits_from_torch_state(params, X_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_array = np.asarray(X_data, dtype=np.float32)
    model = _build_eval_model_from_state(params, X_array.shape[1], device)
    state = {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in params.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, X_array.shape[0], 1024):
            batch = torch.tensor(X_array[start:start + 1024], dtype=torch.float32, device=device)
            logits.append(model(batch).detach().cpu().numpy())
    return np.vstack(logits)


def _build_eval_model_from_state(params, num_features, device):
    if any(key.startswith("1.") for key in params) and any(key.startswith("4.") for key in params):
        channels, height, width = _infer_image_shape(num_features)
        conv1_out = int(np.asarray(params["1.weight"]).shape[0])
        conv2_out = int(np.asarray(params["4.weight"]).shape[0])
        hidden = int(np.asarray(params["8.weight"]).shape[0])
        n_classes = int(np.asarray(params["10.bias"]).shape[0])
        return torch.nn.Sequential(
            torch.nn.Unflatten(1, (channels, height, width)),
            torch.nn.Conv2d(channels, conv1_out, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(conv2_out * (height // 4) * (width // 4), hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_classes),
        ).to(device)
    raise ValueError("Torch state dictionary does not match a supported CNN architecture.")


def _resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def _init_torch_clients(train_val_data, num_features, classes, device, args, privacy_config):
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
                cnn_channels=_parse_cnn_channels(getattr(args, "torch_cnn_channels", "16,32")),
                cnn_fc=getattr(args, "torch_cnn_fc", 128),
            )
        )
    return clients


def _make_client_privacy_state(clients, args, privacy_config, config):
    state = {}
    for idx, client in enumerate(clients):
        n_samples = max(1, len(client.y_train))
        batch_size = min(privacy_config.dp_batch_size, n_samples)
        sample_rate = batch_size / n_samples
        epochs = int(config.get("force_client_epochs") or args.client_epochs)
        planned_steps = args.num_rounds * epochs * int(np.ceil(n_samples / batch_size))
        base_sigma = calibrate_noise_multiplier(
            sample_rate,
            planned_steps,
            privacy_config.epsilon_per_client_total,
            privacy_config.dp_delta,
        )
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


def _run_single_torch_method(method_name, args, train_val_data, X_test, y_test, classes, failure_plan, output_dir, device, privacy_config):
    config = METHOD_CONFIGS[method_name]
    num_features = X_test.shape[1]
    classes = np.asarray(classes, dtype=int)
    clients = _init_torch_clients(train_val_data, num_features, classes, device, args, privacy_config)
    client_ids = [client.client_id for client in clients]
    capabilities = _assign_capabilities(len(clients))
    server = FLServer(f"{method_name}_server", client_ids, num_features, classes=classes)
    server.global_model_parameters = (
        {key: np.copy(value) for key, value in clients[0].model_parameters().items()}
        if clients and getattr(args, "torch_model", "linear") != "linear"
        else _initial_params(num_features, len(classes))
    )
    if config["use_ebcd"]:
        server.ebcd.establish_baseline([client.model_parameters() for client in clients])

    active_indices = list(range(len(clients)))
    current_allocations = _allocate_budget(clients, active_indices, capabilities, config["dynamic_privacy"], privacy_config)
    client_privacy_state = _make_client_privacy_state(clients, args, privacy_config, config) if config["dp_scope"] == "client" else {}
    participation_counts = [0 for _ in clients]
    contribution_history = []
    server_optimizer_state = {} if config.get("aggregation") == "fedadam" else None
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
        "per_client_noise_multiplier": [],
        "privacy_accounting_records": [],
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
        round_noise_multipliers = []
        round_noise_scales = []
        round_delta_norm = 0.0

        for idx, client in enumerate(clients):
            if failure_plan[round_num - 1][idx]:
                round_update_norms.append(None)
                round_ebcd_stats.append((None, None, None))
                round_zkip_status.append(None)
                round_epsilons.append(client_privacy_state[idx]["accountant"].epsilon if idx in client_privacy_state else None)
                round_noise_multipliers.append(None)
                continue
            participation_counts[idx] += 1

            base_epsilon = current_allocations.get(idx, privacy_config.dp_epsilon)
            effective_epsilon = _adjust_epsilon_for_compute(
                base_epsilon, idx, capabilities, config["compute_adapter"], privacy_config
            )
            client.dp_epsilon = max(privacy_config.min_epsilon, min(privacy_config.max_epsilon, effective_epsilon))
            effective_epochs = _effective_epochs(
                idx, args.client_epochs, capabilities, config["compute_adapter"], privacy_config
            )
            if config.get("force_client_epochs") is not None:
                effective_epochs = int(config["force_client_epochs"])
            privacy_state = client_privacy_state.get(idx)
            noise_multiplier = None
            if privacy_state is not None:
                noise_multiplier = (
                    _apdp_noise_multiplier(idx, capabilities, participation_counts, privacy_state, privacy_config, round_num)
                    if config["dynamic_privacy"]
                    else privacy_state["base_noise_multiplier"]
                )
            round_noise_multipliers.append(noise_multiplier)
            delta, proof = client.train(
                global_params=global_params,
                epochs=effective_epochs,
                use_dp=config["dp_scope"] == "client",
                fedprox_mu=args.fedprox_mu if config["fedprox"] else 0.0,
                privacy_accountant=privacy_state["accountant"] if privacy_state is not None else None,
                noise_multiplier=noise_multiplier,
                round_num=round_num,
            )
            if privacy_state is not None and client.last_privacy_event is not None:
                event = client.last_privacy_event
                result["privacy_accounting_records"].append({
                    "method": method_name,
                    "client_id": client.client_id,
                    "round": round_num,
                    "steps": event.steps,
                    "sample_rate": event.sample_rate,
                    "noise_multiplier": event.noise_multiplier,
                    "cumulative_epsilon": event.epsilon,
                    "incremental_epsilon": event.incremental_epsilon,
                    "target_epsilon": privacy_config.epsilon_per_client_total,
                    "remaining_epsilon": privacy_state["accountant"].remaining_epsilon,
                    "status": event.status,
                })
            round_epsilons.append(privacy_state["accountant"].epsilon if privacy_state is not None else None)
            if delta is None:
                round_update_norms.append(None)
                round_ebcd_stats.append((None, None, None))
                round_zkip_status.append(False)
                continue

            update_norm = np.sqrt(sum(np.linalg.norm(v.flatten()) ** 2 for v in delta.values()))
            round_update_norms.append(update_norm)
            round_delta_norm += update_norm
            flat = _flatten_params(delta)
            round_ebcd_stats.append((np.var(flat), float(kurtosis(flat, fisher=True)), float(skew(flat))))
            zkip_ok = server.zkip.verify_proof(delta, proof) if config["use_zkip"] else True
            round_zkip_status.append(zkip_ok)
            local_steps = _local_step_count(len(client.y_train), effective_epochs, privacy_config.dp_batch_size)
            client_updates.append((delta, proof, client.client_id, len(client.y_train), len(client.y_train), local_steps))
            if noise_multiplier is not None:
                noise_stddev = noise_multiplier * client.dp_l2_norm_clip / max(1, min(privacy_config.dp_batch_size, len(client.y_train)))
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
            config=config,
            optimizer_state=server_optimizer_state,
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
        result["per_client_noise_multiplier"].append(round_noise_multipliers)
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
    print(
        f"Torch model: {getattr(args, 'torch_model', 'linear')}; "
        f"mlp_hidden={getattr(args, 'torch_mlp_hidden', '256,128')}; "
        f"cnn_channels={getattr(args, 'torch_cnn_channels', '16,32')}; "
        f"cnn_fc={getattr(args, 'torch_cnn_fc', 128)}"
    )
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

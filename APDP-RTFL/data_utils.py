import os
import pickle
import struct
from collections import Counter

import numpy as np
from sklearn.preprocessing import StandardScaler


PROJECT_DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SUPPORTED_DATASETS = (
    "emnist",
    "synthetic",
    "cifar10",
    "cifar100",
    "medmnist",
    "femnist",
    "chestxray",
    "nih_xray",
    "shakespeare",
)
EMNIST_SPLITS = ("byclass", "balanced", "bymerge", "digits", "letters", "mnist")
PRESPLIT_DATASETS = ("synthetic", "cifar10", "cifar100", "medmnist", "femnist", "chestxray", "nih_xray")


def _to_numpy_features(data):
    if hasattr(data, "numpy"):
        data = data.numpy()
    arr = np.asarray(data)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(float)


def _standardize_train_test(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(np.asarray(X_train, dtype=float))
    X_test = scaler.transform(np.asarray(X_test, dtype=float))
    return X_train.astype(float), X_test.astype(float), scaler


def _print_dataset_summary(name, y_train, y_test, num_features, pre_split_clients=None):
    y_all = np.concatenate([np.asarray(y_train), np.asarray(y_test)])
    class_counts = Counter(y_all.astype(int).tolist())
    print(f"\n=== Dataset Summary: {name} ===")
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    print(f"Features: {num_features}, Classes: {len(class_counts)}")
    if pre_split_clients is not None:
        print(f"Pre-split federated clients: {len(pre_split_clients)}")
    print(f"Class distribution: {dict(sorted(class_counts.items()))}")


def _read_idx_images(path):
    with open(path, "rb") as f:
        magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file.")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n_images, rows * cols).astype(float) / 255.0


def _read_idx_labels(path):
    with open(path, "rb") as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file.")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if len(labels) != n_labels:
        raise ValueError(f"{path} label count mismatch.")
    return labels.astype(int)


def _sample_train_test(X_train, y_train, X_test, y_test, max_samples, random_state):
    if max_samples is None or max_samples <= 0:
        return X_train, y_train, X_test, y_test

    rng = np.random.default_rng(random_state)
    n_train = max(1, int(max_samples * 0.8))
    n_test = max(1, max_samples - n_train)
    train_idx = rng.choice(len(y_train), size=min(n_train, len(y_train)), replace=False)
    test_idx = rng.choice(len(y_test), size=min(n_test, len(y_test)), replace=False)
    return X_train[train_idx], y_train[train_idx], X_test[test_idx], y_test[test_idx]


def load_emnist_idx(data_root=None, random_state=42, max_samples=None, split="byclass"):
    if split not in EMNIST_SPLITS:
        raise ValueError(f"Unsupported EMNIST split '{split}'. Use one of: {', '.join(EMNIST_SPLITS)}.")

    data_root = data_root or PROJECT_DATA_ROOT
    raw_dir = os.path.join(data_root, "emnist", "raw_data", "EMNIST", "raw")
    train_images = os.path.join(raw_dir, f"emnist-{split}-train-images-idx3-ubyte")
    train_labels = os.path.join(raw_dir, f"emnist-{split}-train-labels-idx1-ubyte")
    test_images = os.path.join(raw_dir, f"emnist-{split}-test-images-idx3-ubyte")
    test_labels = os.path.join(raw_dir, f"emnist-{split}-test-labels-idx1-ubyte")

    required = [train_images, train_labels, test_images, test_labels]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "EMNIST IDX files were not found in the project data directory. Missing: "
            + ", ".join(missing)
        )

    X_train = _read_idx_images(train_images)
    y_train = _read_idx_labels(train_labels)
    X_test = _read_idx_images(test_images)
    y_test = _read_idx_labels(test_labels)
    X_train, y_train, X_test, y_test = _sample_train_test(
        X_train, y_train, X_test, y_test, max_samples, random_state
    )
    X_train, X_test, _ = _standardize_train_test(X_train, X_test)
    feature_names = [f"pixel_{i}" for i in range(X_train.shape[1])]
    classes = np.unique(np.concatenate([y_train, y_test]))
    return X_train, y_train, X_test, y_test, feature_names, classes, None


def _read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _read_task_file(path):
    if path.endswith(".pkl"):
        data = _read_pickle(path)
    elif path.endswith(".pt"):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Reading FEMNIST .pt files requires torch to be installed.") from exc
        data = torch.load(path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported FedGreen task file: {path}")

    if isinstance(data, tuple) and len(data) == 2:
        X, y = data
        return _to_numpy_features(X), np.asarray(y).astype(int)

    if isinstance(data, list) and data and isinstance(data[0], tuple) and len(data[0]) == 2:
        X, y = zip(*data)
        return _to_numpy_features(np.asarray(X)), np.asarray(y).astype(int)

    if isinstance(data, (list, np.ndarray)) and len(data) > 0 and np.issubdtype(np.asarray(data).dtype, np.integer):
        raise ValueError(
            f"{path} contains dataset indices, not feature-label pairs. "
            "Run the dataset-specific raw loader or regenerate all_data as materialized samples."
        )

    raise ValueError(f"Could not interpret FedGreen task file: {path}")


def _find_task_file(task_dir, split_name):
    candidates = [
        os.path.join(task_dir, f"{split_name}.pkl"),
        os.path.join(task_dir, f"{split_name}.pt"),
        os.path.join(task_dir, f"{split_name}.txt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_task_split(tasks_root, split_name):
    if not os.path.isdir(tasks_root):
        return [], [], []

    client_data = []
    X_parts = []
    y_parts = []
    task_dirs = [
        os.path.join(tasks_root, name)
        for name in sorted(os.listdir(tasks_root))
        if os.path.isdir(os.path.join(tasks_root, name))
    ]
    for task_dir in task_dirs:
        path = _find_task_file(task_dir, split_name)
        if path is None:
            continue
        X_task, y_task = _read_task_file(path)
        if split_name == "train":
            client_data.append((X_task, y_task))
        X_parts.append(X_task)
        y_parts.append(y_task)
    return client_data, X_parts, y_parts


def load_presplit_all_data(dataset_name, data_root=None):
    if dataset_name == "shakespeare":
        raise NotImplementedError(
            "The current APDP-RTFL classifier uses SGDClassifier and does not support Shakespeare sequence modeling."
        )

    data_root = data_root or PROJECT_DATA_ROOT
    all_data_dir = os.path.join(data_root, dataset_name, "all_data")
    if not os.path.isdir(all_data_dir):
        raise FileNotFoundError(
            f"{all_data_dir} does not exist. Generate FedGreen-style client data first, for example by running "
            f"data/{dataset_name}/generate_data.py with the required arguments."
        )

    train_root = os.path.join(all_data_dir, "train")
    test_root = os.path.join(all_data_dir, "test")
    client_data, X_train_parts, y_train_parts = _load_task_split(train_root, "train")
    _, X_test_parts, y_test_parts = _load_task_split(train_root, "test")
    _, X_external_test_parts, y_external_test_parts = _load_task_split(test_root, "test")
    X_test_parts.extend(X_external_test_parts)
    y_test_parts.extend(y_external_test_parts)

    if not X_train_parts:
        raise FileNotFoundError(f"No train task files were found under {train_root}.")
    if not X_test_parts:
        raise FileNotFoundError(f"No test task files were found under {all_data_dir}.")

    X_train = np.concatenate(X_train_parts)
    y_train = np.concatenate(y_train_parts).astype(int)
    X_test = np.concatenate(X_test_parts)
    y_test = np.concatenate(y_test_parts).astype(int)
    X_train, X_test, scaler = _standardize_train_test(X_train, X_test)
    standardized_clients = [
        (scaler.transform(np.asarray(X_client, dtype=float)), y_client.astype(int))
        for X_client, y_client in client_data
    ]
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    classes = np.unique(np.concatenate([y_train, y_test]))
    return X_train, y_train, X_test, y_test, feature_names, classes, standardized_clients


def load_experiment_data(
    dataset_name="emnist",
    data_root=None,
    random_state=42,
    max_samples=None,
    emnist_split="byclass",
):
    dataset_name = dataset_name.lower()
    if dataset_name == "nih_xray":
        dataset_name = "NIH_Xray"
    elif dataset_name == "chestxray":
        dataset_name = "Chestxray"

    if dataset_name.lower() not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Use one of: {', '.join(SUPPORTED_DATASETS)}.")

    if dataset_name.lower() == "emnist":
        result = load_emnist_idx(
            data_root=data_root,
            random_state=random_state,
            max_samples=max_samples,
            split=emnist_split,
        )
    else:
        result = load_presplit_all_data(dataset_name, data_root=data_root)

    X_train, y_train, X_test, y_test, feature_names, classes, client_data = result
    _print_dataset_summary(dataset_name, y_train, y_test, X_train.shape[1], pre_split_clients=client_data)
    return result


def _client_split_sizes(n_total, num_clients, size_ratios=None, min_samples=1):
    if size_ratios is None:
        weights = np.ones(num_clients) / num_clients
    else:
        weights = np.asarray(size_ratios, dtype=float)
        if len(weights) != num_clients:
            raise ValueError("size_ratios length must equal num_clients.")
        weights = weights / weights.sum()

    raw_sizes = np.floor(weights * n_total).astype(int)
    raw_sizes = np.maximum(raw_sizes, min_samples)
    while raw_sizes.sum() > n_total:
        idx = int(np.argmax(raw_sizes))
        raw_sizes[idx] -= 1
    while raw_sizes.sum() < n_total:
        idx = int(np.argmin(raw_sizes))
        raw_sizes[idx] += 1
    return raw_sizes.tolist()


def split_data_for_clients(
    X_train_full,
    y_train_full,
    num_clients,
    size_ratios=None,
    min_samples=1,
    partition="dirichlet",
    dirichlet_alpha=0.5,
    random_state=42,
):
    X_np = np.asarray(X_train_full)
    y_np = np.asarray(y_train_full).astype(int)
    rng = np.random.default_rng(random_state)
    n_total = len(y_np)

    if n_total < num_clients:
        raise ValueError("Number of samples must be at least num_clients.")

    partition = partition.lower()
    split_sizes = _client_split_sizes(n_total, num_clients, size_ratios, min_samples)
    client_indices = [[] for _ in range(num_clients)]

    if partition in {"iid", "quantity_skew", "quantity"}:
        shuffled = rng.permutation(n_total)
        start = 0
        for client_id, size in enumerate(split_sizes):
            client_indices[client_id] = shuffled[start:start + size].tolist()
            start += size
    elif partition in {"dirichlet", "label_skew", "non_iid"}:
        classes = np.unique(y_np)
        for cls in classes:
            cls_idx = np.where(y_np == cls)[0]
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(np.repeat(dirichlet_alpha, num_clients))
            proportions = proportions / proportions.sum()
            cuts = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
            for client_id, part in enumerate(np.split(cls_idx, cuts)):
                client_indices[client_id].extend(part.tolist())

        empty_clients = [i for i, idx in enumerate(client_indices) if len(idx) == 0]
        for empty_id in empty_clients:
            donor_id = int(np.argmax([len(idx) for idx in client_indices]))
            moved = client_indices[donor_id][-1:]
            client_indices[donor_id] = client_indices[donor_id][:-1]
            client_indices[empty_id].extend(moved)
    else:
        raise ValueError("partition must be one of iid, quantity_skew, dirichlet, label_skew, or non_iid.")

    client_data = []
    print("\nClient Data Distribution:")
    for client_id, indices in enumerate(client_indices):
        indices = np.asarray(indices, dtype=int)
        rng.shuffle(indices)
        X_c = X_np[indices]
        y_c = y_np[indices]
        counts = dict(sorted(Counter(y_c.tolist()).items()))
        print(f"Client {client_id}: size={len(y_c)}, classes={counts}")
        client_data.append((X_c, y_c))
    return client_data

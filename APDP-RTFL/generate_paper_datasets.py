"""Generate materialized federated datasets for GRAIL-FL paper experiments.

The legacy dataset scripts in data/* often save sample indices.  The current
experiment loader expects each client task to contain feature-label pairs, so
this script writes train/test.pkl files directly as (X, y) tuples.
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split


PROJECT_DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_DATASETS = ("femnist", "cifar10", "medmnist")


def _save_pair(path, X, y):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump((np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)), handle)


def _ensure_new_output_dir(path, overwrite):
    if os.path.exists(path):
        if overwrite:
            raise FileExistsError(
                f"{path} already exists. Bulk deletion is disabled by project policy; "
                "please remove or archive that directory manually before regenerating."
            )
        raise FileExistsError(f"{path} already exists. Use the existing data or move it manually before regenerating.")
    os.makedirs(path, exist_ok=True)


def _flatten_features(X):
    X = np.asarray(X)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    return X.astype(np.float32)


def _stratified_client_indices(y, n_clients, alpha, seed, min_size):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int64)
    classes = np.unique(y)
    for attempt in range(100):
        client_indices = [[] for _ in range(n_clients)]
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(np.repeat(alpha, n_clients))
            cuts = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
            for client_id, part in enumerate(np.split(cls_idx, cuts)):
                client_indices[client_id].extend(part.tolist())
        sizes = [len(idx) for idx in client_indices]
        if min(sizes) >= min_size:
            return [np.asarray(idx, dtype=np.int64) for idx in client_indices]
        alpha = min(alpha * 1.25, 10.0)
    raise RuntimeError("Could not create non-empty client partitions; increase samples or alpha.")


def _write_federated_split(dataset_name, X, y, args):
    out_root = os.path.join(args.data_root, dataset_name, "all_data")
    _ensure_new_output_dir(out_root, args.overwrite)
    indices_by_client = _stratified_client_indices(
        y, args.n_tasks, args.alpha, args.seed, args.min_client_samples
    )
    summary = []
    for client_id, indices in enumerate(indices_by_client):
        y_client = y[indices]
        stratify = y_client if len(np.unique(y_client)) > 1 and min(Counter(y_client).values()) >= 2 else None
        train_idx, test_idx = train_test_split(
            indices,
            train_size=args.tr_frac,
            random_state=args.seed,
            stratify=stratify,
        )
        task_dir = os.path.join(out_root, "train", f"task_{client_id}")
        _save_pair(os.path.join(task_dir, "train.pkl"), X[train_idx], y[train_idx])
        _save_pair(os.path.join(task_dir, "test.pkl"), X[test_idx], y[test_idx])
        counts = Counter(y_client.tolist())
        summary.append(
            {
                "task": client_id,
                "total": len(indices),
                "train": len(train_idx),
                "test": len(test_idx),
                "classes": len(counts),
            }
        )
    with open(os.path.join(out_root, "dataset_summary.csv"), "w", encoding="utf-8") as handle:
        handle.write("task,total,train,test,classes\n")
        for row in summary:
            handle.write("{task},{total},{train},{test},{classes}\n".format(**row))
    print(f"[ok] {dataset_name}: wrote {len(summary)} clients to {out_root}")


def _write_client_tasks(dataset_name, client_pairs, args):
    out_root = os.path.join(args.data_root, dataset_name, "all_data")
    _ensure_new_output_dir(out_root, args.overwrite)
    rng = np.random.default_rng(args.seed)
    summary = []
    for client_id, (X_client, y_client) in enumerate(client_pairs):
        X_client = _flatten_features(X_client)
        y_client = np.asarray(y_client, dtype=np.int64)
        if len(y_client) < args.min_client_samples:
            continue
        indices = np.arange(len(y_client))
        rng.shuffle(indices)
        stratify = y_client if len(np.unique(y_client)) > 1 and min(Counter(y_client).values()) >= 2 else None
        train_idx, test_idx = train_test_split(
            indices,
            train_size=args.tr_frac,
            random_state=args.seed,
            stratify=stratify,
        )
        task_dir = os.path.join(out_root, "train", f"task_{len(summary)}")
        _save_pair(os.path.join(task_dir, "train.pkl"), X_client[train_idx], y_client[train_idx])
        _save_pair(os.path.join(task_dir, "test.pkl"), X_client[test_idx], y_client[test_idx])
        summary.append(
            {
                "task": len(summary),
                "source_client": client_id,
                "total": len(y_client),
                "train": len(train_idx),
                "test": len(test_idx),
                "classes": len(np.unique(y_client)),
            }
        )
        if len(summary) >= args.n_tasks:
            break
    if len(summary) < args.n_tasks:
        raise RuntimeError(f"Only {len(summary)} FEMNIST writer clients met the minimum sample requirement.")
    with open(os.path.join(out_root, "dataset_summary.csv"), "w", encoding="utf-8") as handle:
        handle.write("task,source_client,total,train,test,classes\n")
        for row in summary:
            handle.write("{task},{source_client},{total},{train},{test},{classes}\n".format(**row))
    print(f"[ok] {dataset_name}: wrote {len(summary)} writer clients to {out_root}")


def _load_cifar10(data_root):
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor

    raw_root = os.path.join(data_root, "cifar10", "raw_data")
    train = CIFAR10(root=raw_root, train=True, download=True, transform=ToTensor())
    test = CIFAR10(root=raw_root, train=False, download=True, transform=ToTensor())
    X, y = [], []
    for dataset in (train, test):
        for image, label in dataset:
            X.append(image.numpy())
            y.append(label)
    return _flatten_features(np.asarray(X)), np.asarray(y, dtype=np.int64)


def _load_medmnist(data_root, flag):
    npz_path = os.path.join(data_root, "medmnist", "raw_data", f"{flag}.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"{npz_path} does not exist.")
    data = np.load(npz_path)
    X = np.concatenate([data["train_images"], data["val_images"], data["test_images"]])
    y = np.concatenate([data["train_labels"], data["val_labels"], data["test_labels"]]).reshape(-1)
    return _flatten_features(X.astype(np.float32) / 255.0), np.asarray(y, dtype=np.int64)


def _load_femnist(data_root, s_frac, seed):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("FEMNIST materialization requires torch and LEAF writer tensors.") from exc

    writer_root = os.path.join(data_root, "femnist", "intermediate", "data_as_tensor_by_writer")
    if not os.path.isdir(writer_root):
        raise FileNotFoundError(
            f"{writer_root} does not exist. Run data/femnist/preprocess.sh first to create "
            "LEAF FEMNIST writer tensors, then rerun this script for --datasets femnist."
        )
    files = sorted(
        os.path.join(writer_root, name)
        for name in os.listdir(writer_root)
        if name.endswith((".pt", ".pkl"))
    )
    if not files:
        raise FileNotFoundError(f"No writer tensor files found under {writer_root}.")
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    keep = max(1, int(len(files) * s_frac))
    client_pairs = []
    for path in files[:keep]:
        if path.endswith(".pt"):
            data, targets = torch.load(path, map_location="cpu")
            if hasattr(data, "numpy"):
                data = data.numpy()
            if hasattr(targets, "numpy"):
                targets = targets.numpy()
        else:
            with open(path, "rb") as handle:
                data, targets = pickle.load(handle)
        client_pairs.append((_flatten_features(data), np.asarray(targets, dtype=np.int64)))
    return client_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Generate materialized paper datasets for GRAIL-FL.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--data-root", default=PROJECT_DATA_ROOT)
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tr-frac", type=float, default=0.8)
    parser.add_argument("--s-frac", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-client-samples", type=int, default=10)
    parser.add_argument("--medmnist-flag", default="pathmnist")
    parser.add_argument("--overwrite", action="store_true", help="Accepted for compatibility, but never deletes data.")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    for dataset in datasets:
        if dataset == "cifar10":
            X, y = _load_cifar10(args.data_root)
        elif dataset == "medmnist":
            X, y = _load_medmnist(args.data_root, args.medmnist_flag)
        elif dataset == "femnist":
            client_pairs = _load_femnist(args.data_root, args.s_frac, args.seed)
            print(f"{dataset}: writer_clients={len(client_pairs)}")
            _write_client_tasks(dataset, client_pairs, args)
            continue
        else:
            raise ValueError(f"Unsupported generator dataset: {dataset}")
        print(f"{dataset}: samples={len(y)}, features={X.shape[1]}, classes={len(np.unique(y))}")
        _write_federated_split(dataset, X, y, args)


if __name__ == "__main__":
    main()

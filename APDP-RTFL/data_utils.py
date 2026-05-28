import os
import struct
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


DATA_FILE_APPLICATION = "application_record.csv"
DATA_FILE_CREDIT = "credit_record.csv"
FEDGREEN_DATA_ROOT = r"C:\Users\Hao\Desktop\FedGreen工程\1.测试\data"


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _to_numpy_features(data):
    if hasattr(data, "numpy"):
        data = data.numpy()
    arr = np.asarray(data)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(float)


def _standardize_train_test(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train.astype(float), X_test.astype(float)


def _print_dataset_summary(name, y_train, y_test, num_features):
    y_all = np.concatenate([np.asarray(y_train), np.asarray(y_test)])
    class_counts = Counter(y_all.tolist())
    print(f"\n=== Dataset Summary: {name} ===")
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    print(f"Features: {num_features}, Classes: {len(class_counts)}")
    print(f"Class distribution: {dict(sorted(class_counts.items()))}")


def load_credit_card_data(random_state=42):
    try:
        app_df = pd.read_csv(DATA_FILE_APPLICATION)
        cred_df = pd.read_csv(DATA_FILE_CREDIT)
    except FileNotFoundError:
        print(f"Error: Ensure '{DATA_FILE_APPLICATION}' and '{DATA_FILE_CREDIT}' are in the current directory.")
        print("Download from: https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction")
        return None, None, None, None, None, None

    df = pd.merge(app_df, cred_df, on="ID", how="inner")

    def determine_credit_risk(group):
        if any(s in ["2", "3", "4", "5"] for s in group["STATUS"].astype(str)):
            return 1
        return 0

    target_df = df.groupby("ID").apply(determine_credit_risk).reset_index(name="TARGET")
    unique_app_df = df.drop_duplicates(subset=["ID"], keep="first")
    df = pd.merge(unique_app_df, target_df, on="ID", how="left")

    df = df.drop(["ID", "MONTHS_BALANCE", "STATUS", "FLAG_MOBIL"], axis=1)
    df = df.dropna(subset=["TARGET"])

    for col in df.select_dtypes(include="object").columns:
        mode = df[col].mode()
        df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    df["DAYS_BIRTH"] = np.abs(df["DAYS_BIRTH"]) / 365
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].apply(lambda x: 0 if x > 0 else np.abs(x) / 365)

    X = df.drop("TARGET", axis=1)
    y = df["TARGET"].astype(int).to_numpy()

    categorical_features = X.select_dtypes(include="object").columns
    numerical_features = X.select_dtypes(include=np.number).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", _make_one_hot_encoder(), categorical_features),
        ],
        remainder="passthrough",
    )

    X_processed = preprocessor.fit_transform(X).astype(float)
    try:
        feature_names_cat = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
    except AttributeError:
        feature_names_cat = []
    feature_names = numerical_features.tolist() + list(feature_names_cat)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=random_state, stratify=y
    )
    return X_train, y_train, X_test, y_test, feature_names, np.array([0, 1])


def load_sklearn_dataset(dataset_name, test_size=0.2, random_state=42):
    if dataset_name == "digits":
        bundle = load_digits()
    elif dataset_name == "breast_cancer":
        bundle = load_breast_cancer()
    elif dataset_name == "wine":
        bundle = load_wine()
    else:
        raise ValueError(f"Unsupported sklearn dataset: {dataset_name}")

    X = bundle.data.astype(float)
    y = bundle.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test = _standardize_train_test(X_train, X_test)
    feature_names = [str(name) for name in getattr(bundle, "feature_names", [f"feature_{i}" for i in range(X.shape[1])])]
    classes = np.unique(y)
    return X_train, y_train, X_test, y_test, feature_names, classes


def load_torchvision_dataset(dataset_name, data_root=None, test_size=0.2, random_state=42, download=False, max_samples=None):
    try:
        from torchvision.datasets import CIFAR10, EMNIST, FashionMNIST, MNIST
    except ImportError as exc:
        if dataset_name == "emnist":
            return load_emnist_idx_from_fedgreen(data_root, random_state=random_state, max_samples=max_samples)
        raise ImportError("torchvision is required for MNIST/Fashion-MNIST/EMNIST/CIFAR-10 experiments.") from exc

    data_root = data_root or FEDGREEN_DATA_ROOT
    dataset_roots = {
        "mnist": os.path.join(data_root, "mnist", "raw_data"),
        "fashion_mnist": os.path.join(data_root, "fashion_mnist", "raw_data"),
        "emnist": os.path.join(data_root, "emnist", "raw_data"),
        "cifar10": os.path.join(data_root, "cifar10", "raw_data"),
    }
    root = dataset_roots[dataset_name]

    if dataset_name == "mnist":
        train_ds = MNIST(root=root, train=True, download=download)
        test_ds = MNIST(root=root, train=False, download=download)
        X = np.concatenate([_to_numpy_features(train_ds.data), _to_numpy_features(test_ds.data)])
        y = np.concatenate([np.asarray(train_ds.targets), np.asarray(test_ds.targets)])
    elif dataset_name == "fashion_mnist":
        train_ds = FashionMNIST(root=root, train=True, download=download)
        test_ds = FashionMNIST(root=root, train=False, download=download)
        X = np.concatenate([_to_numpy_features(train_ds.data), _to_numpy_features(test_ds.data)])
        y = np.concatenate([np.asarray(train_ds.targets), np.asarray(test_ds.targets)])
    elif dataset_name == "emnist":
        train_ds = EMNIST(root=root, split="byclass", train=True, download=download)
        test_ds = EMNIST(root=root, split="byclass", train=False, download=download)
        X = np.concatenate([_to_numpy_features(train_ds.data), _to_numpy_features(test_ds.data)])
        y = np.concatenate([np.asarray(train_ds.targets), np.asarray(test_ds.targets)])
    elif dataset_name == "cifar10":
        train_ds = CIFAR10(root=root, train=True, download=download)
        test_ds = CIFAR10(root=root, train=False, download=download)
        X = np.concatenate([_to_numpy_features(train_ds.data), _to_numpy_features(test_ds.data)])
        y = np.concatenate([np.asarray(train_ds.targets), np.asarray(test_ds.targets)])
    else:
        raise ValueError(f"Unsupported torchvision dataset: {dataset_name}")

    if max_samples is not None and max_samples > 0 and len(y) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(y), size=max_samples, replace=False)
        X, y = X[idx], y[idx]

    X = X / 255.0 if np.nanmax(X) > 1.0 else X
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.astype(int), test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test = _standardize_train_test(X_train, X_test)
    feature_names = [f"pixel_{i}" for i in range(X_train.shape[1])]
    classes = np.unique(y)
    return X_train, y_train, X_test, y_test, feature_names, classes


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


def load_emnist_idx_from_fedgreen(data_root=None, random_state=42, max_samples=None, split="byclass"):
    data_root = data_root or FEDGREEN_DATA_ROOT
    raw_dir = os.path.join(data_root, "emnist", "raw_data", "EMNIST", "raw")
    train_images = os.path.join(raw_dir, f"emnist-{split}-train-images-idx3-ubyte")
    train_labels = os.path.join(raw_dir, f"emnist-{split}-train-labels-idx1-ubyte")
    test_images = os.path.join(raw_dir, f"emnist-{split}-test-images-idx3-ubyte")
    test_labels = os.path.join(raw_dir, f"emnist-{split}-test-labels-idx1-ubyte")

    required = [train_images, train_labels, test_images, test_labels]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "EMNIST IDX files were not found under the FedGreen data root. Missing: "
            + ", ".join(missing)
        )

    X_train = _read_idx_images(train_images)
    y_train = _read_idx_labels(train_labels)
    X_test = _read_idx_images(test_images)
    y_test = _read_idx_labels(test_labels)

    if max_samples is not None and max_samples > 0:
        rng = np.random.default_rng(random_state)
        n_train = max(1, int(max_samples * 0.8))
        n_test = max(1, max_samples - n_train)
        train_idx = rng.choice(len(y_train), size=min(n_train, len(y_train)), replace=False)
        test_idx = rng.choice(len(y_test), size=min(n_test, len(y_test)), replace=False)
        X_train, y_train = X_train[train_idx], y_train[train_idx]
        X_test, y_test = X_test[test_idx], y_test[test_idx]

    X_train, X_test = _standardize_train_test(X_train, X_test)
    feature_names = [f"pixel_{i}" for i in range(X_train.shape[1])]
    classes = np.unique(np.concatenate([y_train, y_test]))
    return X_train, y_train, X_test, y_test, feature_names, classes


def load_and_preprocess_data(
    dataset_name="credit_card",
    data_root=None,
    test_size=0.2,
    random_state=42,
    download=False,
    max_samples=None,
):
    dataset_name = dataset_name.lower()
    if dataset_name in {"credit", "credit_card", "credit_card_approval"}:
        result = load_credit_card_data(random_state=random_state)
    elif dataset_name in {"digits", "breast_cancer", "wine"}:
        result = load_sklearn_dataset(dataset_name, test_size=test_size, random_state=random_state)
    elif dataset_name in {"mnist", "fashion_mnist", "emnist", "cifar10"}:
        result = load_torchvision_dataset(
            dataset_name,
            data_root=data_root,
            test_size=test_size,
            random_state=random_state,
            download=download,
            max_samples=max_samples,
        )
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            "Use credit_card, digits, breast_cancer, wine, mnist, fashion_mnist, emnist, or cifar10."
        )

    X_train, y_train, X_test, y_test, feature_names, classes = result
    if X_train is not None:
        _print_dataset_summary(dataset_name, y_train, y_test, X_train.shape[1])
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

    if partition == "iid":
        shuffled = rng.permutation(n_total)
        start = 0
        for client_id, size in enumerate(split_sizes):
            client_indices[client_id] = shuffled[start:start + size].tolist()
            start += size
    elif partition in {"quantity_skew", "quantity"}:
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

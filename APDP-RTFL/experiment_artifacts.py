"""Shared persistence helpers for reproducible APDP-RTFL experiment artifacts."""

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def json_safe(value):
    """Expose JSON-safe conversion for run metadata written by training entry points."""
    return _json_safe(value)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)


def _package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_revision(repo_root):
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def initialize_run_artifacts(output_dir, args):
    """Write immutable run metadata before data loading and training begin."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    command = subprocess.list2cmdline(sys.argv)
    _write_json(
        os.path.join(output_dir, "run_config.json"),
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_directory": os.path.abspath(output_dir),
            "arguments": vars(args).copy(),
            "git_revision": _git_revision(repo_root),
        },
    )
    with open(os.path.join(output_dir, "run_command.txt"), "w", encoding="utf-8") as handle:
        handle.write(command + "\n")
    _write_json(
        os.path.join(output_dir, "environment.json"),
        {
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": {
                package: _package_version(package)
                for package in ("numpy", "scipy", "scikit-learn", "matplotlib", "torch")
            },
        },
    )


def _array_fingerprint(array):
    values = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("utf-8"))
    digest.update(str(values.dtype).encode("utf-8"))
    view = memoryview(values).cast("B")
    chunk_size = 4 * 1024 * 1024
    for offset in range(0, len(view), chunk_size):
        digest.update(view[offset:offset + chunk_size])
    return digest.hexdigest()


def _class_distribution(labels):
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(classes, counts)}


def _write_csv(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["name"])
        writer.writeheader()
        writer.writerows(rows)


def write_data_artifacts(output_dir, args, train_val_data, X_test, y_test, classes, failure_plan, dataset_name=None, metadata_rows=None):
    """Persist dataset fingerprints, client split summaries, and the generated failure plan."""
    dataset_name = dataset_name or args.dataset
    root = os.path.join(output_dir, "data_artifacts")
    if dataset_name != args.dataset or metadata_rows is not None:
        root = os.path.join(root, dataset_name)
    os.makedirs(root, exist_ok=True)
    manifest_path = os.path.join(root, "dataset_manifest.json")
    if os.path.exists(manifest_path):
        return

    manifest = {
        "dataset": dataset_name,
        "emnist_split": getattr(args, "emnist_split", None),
        "partition": getattr(args, "partition", None),
        "dirichlet_alpha": getattr(args, "dirichlet_alpha", None),
        "seed": getattr(args, "seed", None),
        "classes": [int(item) for item in np.asarray(classes)],
        "test_samples": int(len(y_test)),
        "test_features": int(np.asarray(X_test).shape[1]) if np.asarray(X_test).ndim > 1 else 1,
        "test_class_distribution": _class_distribution(y_test),
        "test_feature_fingerprint_sha256": _array_fingerprint(X_test),
        "test_label_fingerprint_sha256": _array_fingerprint(y_test),
    }
    _write_json(manifest_path, manifest)

    metadata_by_client = {row.get("client_id"): row for row in metadata_rows or []}
    partition_rows = []
    for client_idx, (X_train, y_train, X_val, y_val) in enumerate(train_val_data):
        client_id = f"client_{client_idx}"
        row = {
            "client_id": client_id,
            "train_samples": int(len(y_train)),
            "validation_samples": int(len(y_val)) if y_val is not None else 0,
            "train_class_distribution": json.dumps(_class_distribution(y_train), ensure_ascii=False, sort_keys=True),
            "validation_class_distribution": json.dumps(_class_distribution(y_val), ensure_ascii=False, sort_keys=True) if y_val is not None else "{}",
            "train_feature_fingerprint_sha256": _array_fingerprint(X_train),
            "train_label_fingerprint_sha256": _array_fingerprint(y_train),
        }
        if X_val is not None and y_val is not None:
            row["validation_feature_fingerprint_sha256"] = _array_fingerprint(X_val)
            row["validation_label_fingerprint_sha256"] = _array_fingerprint(y_val)
        synthetic_metadata = metadata_by_client.get(client_id)
        if synthetic_metadata:
            for key in ("gender_group", "age_group", "region_group", "compute_level", "data_quality"):
                row[key] = synthetic_metadata.get(key, "")
        partition_rows.append(row)
    _write_csv(os.path.join(root, "client_partition_manifest.csv"), partition_rows)
    np.save(os.path.join(root, "failure_plan.npy"), np.asarray(failure_plan, dtype=bool))


def export_tcm_checkpoints(output_dir, tcm):
    """Export TCM hashes and model states without relying on pickle serialization."""
    if not getattr(tcm, "manifold_log", None):
        return
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_files = {}
    for model_hash, parameters in tcm.detailed_states.items():
        file_name = f"model_{model_hash[:16]}.npz"
        np.savez_compressed(os.path.join(checkpoint_dir, file_name), **parameters)
        state_files[model_hash] = os.path.join("checkpoints", file_name)
    rows = []
    for timestamp, round_num, model_hash, server_hash, client_updates_hash in tcm.manifold_log:
        rows.append(
            {
                "timestamp": timestamp,
                "round": round_num,
                "model_hash": model_hash,
                "server_state_hash": server_hash,
                "client_updates_hash": client_updates_hash,
                "checkpoint_file": state_files.get(model_hash, ""),
            }
        )
    _write_csv(os.path.join(output_dir, "tcm_manifest.csv"), rows)


_CHART_SOURCES = {
    "global_metrics.png": ("metrics.csv", "metrics.json"),
    "dp_noise_scale.png": ("metrics.csv",),
    "agg_client_counts.png": ("metrics.csv",),
    "zkip_failures.png": ("metrics.csv",),
    "delta_norm.png": ("metrics.json",),
    "ebcd_alerts.png": ("metrics.csv",),
    "tcm_state_count.png": ("metrics.csv", "tcm_manifest.csv"),
    "per_client_update_norms.png": ("per_client_update_norms.npy",),
    "per_client_ebcd_variance.png": ("per_client_ebcd_stats.npy",),
    "per_client_ebcd_kurtosis.png": ("per_client_ebcd_stats.npy",),
    "per_client_ebcd_skewness.png": ("per_client_ebcd_stats.npy",),
    "per_client_zkip_status.png": ("per_client_zkip_status.npy",),
    "per_client_epsilon_dynamic_dp.png": ("per_client_epsilon.npy",),
    "ebcd_stats.png": ("metrics.csv",),
    "server_status.png": ("metrics.json",),
    "privacy_budget_allocation.png": ("metrics.json",),
    "epsilon_vs_data_quality.png": ("client_privacy_quality_summary.csv",),
    "earlystop_server_best_val_acc.png": ("metrics.json",),
    "regulatory_actions.png": ("regulatory_intervention_summary.csv",),
    "regulatory_risk_by_client.png": ("regulatory_intervention_summary.csv",),
    "client_performance_fairness.png": ("client_fairness_summary.csv", "metrics.json"),
    "privacy_budget_fairness.png": ("client_fairness_summary.csv", "metrics.json"),
    "participation_fairness.png": ("client_fairness_summary.csv", "metrics.json"),
    "contribution_score_by_client.png": ("contribution_penalty_summary.csv",),
    "approx_shapley_by_client.png": ("contribution_penalty_summary.csv",),
    "penalty_components.png": ("contribution_penalty_summary.csv", "metrics.json"),
    "contribution_weight_alignment.png": ("contribution_penalty_summary.csv", "metrics.json"),
    "baseline_comparison.png": ("baseline_summary.csv",),
    "pollution_accuracy.png": ("pollution_summary.csv",),
    "pollution_f1.png": ("pollution_summary.csv",),
    "pollution_regulatory_actions.png": ("pollution_summary.csv",),
    "pollution_detection_rate.png": ("pollution_final_metrics.csv",),
    "audit_trace_timeline.png": ("audit_trace_log.csv", "audit_chain_verification.csv"),
    "ablation_accuracy.png": ("ablation_summary.csv",),
    "ablation_macro_f1.png": ("ablation_summary.csv",),
    "ablation_balanced_accuracy.png": ("ablation_summary.csv",),
    "ablation_accuracy_delta.png": ("ablation_final_metrics.csv",),
    "privacy_budget_accuracy.png": ("privacy_sensitivity_final_metrics.csv",),
    "privacy_budget_noise.png": ("privacy_sensitivity_final_metrics.csv",),
    "privacy_budget_tradeoff.png": ("privacy_sensitivity_final_metrics.csv",),
}


def write_artifact_manifest(output_dir):
    """Record file roles and chart-to-data relationships for the current directory tree."""
    rows = []
    for root, _, filenames in os.walk(output_dir):
        for filename in sorted(filenames):
            relative_path = os.path.relpath(os.path.join(root, filename), output_dir)
            artifact_type = "chart" if filename.endswith(".png") else "data"
            sources = _CHART_SOURCES.get(filename, ())
            rows.append(
                {
                    "relative_path": relative_path,
                    "artifact_type": artifact_type,
                    "source_files": ";".join(sources),
                    "description": "Chart source mapping" if sources else "Experiment artifact",
                }
            )
    _write_csv(os.path.join(output_dir, "artifact_manifest.csv"), rows)

import argparse
import json
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, entropy
from data_utils import EMNIST_SPLITS, PROJECT_DATA_ROOT, SUPPORTED_DATASETS, load_experiment_data, split_data_for_clients
from fl_client import FLClient
from fl_server import FLServer
import charting as charts
from sklearn.model_selection import train_test_split
from experiment_artifacts import (
    export_tcm_checkpoints,
    initialize_run_artifacts,
    json_safe,
    write_artifact_manifest,
    write_data_artifacts,
)

# 新增：动态隐私预算配置
TOTAL_PRIVACY_BUDGET = 5.0 # 总隐私预算
MIN_EPSILON = 0.1 # 最小隐私预算
MAX_EPSILON = 2.0 # 最大隐私预算
PRIVACY_ALLOCATION_STRATEGY = "hybrid" # 分配策略：uniform, data_quality, hybrid

NUM_CLIENTS = 5
NUM_ROUNDS = 20
CLIENT_EPOCHS = 3
BASE_LEARNING_RATE = 0.01
DP_EPSILON = 1.0 # 默认值，会被动态分配覆盖
DP_DELTA = 1e-5
DP_L2_NORM_CLIP = 1.0
EARLYSTOP_PATIENCE = 3


def parse_args():
    parser = argparse.ArgumentParser(description="APDP-RTFL federated learning experiment")
    parser.add_argument("--dataset", default="emnist",
                        choices=list(SUPPORTED_DATASETS),
                        help="实验数据集。默认 emnist，仅从项目 data/ 目录读取。")
    parser.add_argument("--data-root", default=PROJECT_DATA_ROOT,
                        help="项目本地数据根目录，默认指向 APDP-RTFL/data。")
    parser.add_argument("--emnist-split", default="byclass",
                        choices=list(EMNIST_SPLITS),
                        help="EMNIST 子集。默认 byclass，可选 digits/balanced/bymerge/letters/mnist。")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="对 EMNIST 抽样，便于快速调试；正式实验可不设置。")
    parser.add_argument("--num-clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--num-rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--client-epochs", type=int, default=CLIENT_EPOCHS)
    parser.add_argument("--partition", default="dirichlet",
                        choices=["iid", "quantity_skew", "dirichlet", "label_skew", "non_iid"],
                        help="客户端数据划分方式。dirichlet/label_skew 用于模拟 non-IID。")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5,
                        help="Dirichlet non-IID 强度；越小标签偏斜越强。")
    parser.add_argument("--output-root", default="results",
                        help="实验结果根目录。默认保存到 results。")
    parser.add_argument("--run-name", default=None,
                        help="本次实验目录名。默认自动使用 数据集_YYYYmmdd_HHMMSS。")
    parser.add_argument("--experiment-suite", default="single",
                        choices=["single", "baselines", "participation", "privacy_sensitivity", "pollution", "fairness", "synthetic_fairness", "contribution", "audit_trace", "ablation"],
                        help="single runs APDP-RTFL; comparison suites include baselines, participation, privacy_sensitivity, pollution, fairness, synthetic_fairness, contribution, audit_trace, and ablation.")
    parser.add_argument("--methods", default="all",
                        help="Baseline methods: all or comma-separated dp_fl,dp_flprox,dp_fedsgd,global_dp,dp_rtfl,apdp_rtfl. Legacy fedavg/fedprox/ldp_fl remain available when explicitly requested.")
    parser.add_argument("--fedprox-mu", type=float, default=0.01,
                        help="FedProx proximal coefficient. Default: 0.01.")
    parser.add_argument("--backend", default="sklearn", choices=["sklearn", "torch"],
                        help="Training backend. Use torch with --device cuda for GPU experiments.")
    parser.add_argument("--device", default="auto",
                        help="Torch device for --backend torch: auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--torch-batch-size", type=int, default=256,
                        help="Mini-batch size for the torch backend.")
    parser.add_argument("--total-privacy-budget", type=float, default=TOTAL_PRIVACY_BUDGET,
                        help="Total privacy budget allocated across active clients.")
    parser.add_argument("--min-epsilon", type=float, default=MIN_EPSILON,
                        help="Minimum per-client epsilon after adaptive adjustment.")
    parser.add_argument("--max-epsilon", type=float, default=MAX_EPSILON,
                        help="Maximum per-client epsilon after adaptive adjustment.")
    parser.add_argument("--dp-epsilon", type=float, default=DP_EPSILON,
                        help="Fallback per-client epsilon before allocation overrides it.")
    parser.add_argument("--dp-delta", type=float, default=DP_DELTA,
                        help="Differential privacy delta.")
    parser.add_argument("--dp-l2-norm-clip", type=float, default=DP_L2_NORM_CLIP,
                        help="L2 clipping norm for DP noise calibration.")
    parser.add_argument("--failure-prob", type=float, default=0.15,
                        help="Per-round client failure probability. Use 0 for utility-only experiments.")
    parser.add_argument("--apdp-warmup-rounds", type=int, default=20,
                        help="Rounds before APDP adaptive epsilon adjustment starts.")
    parser.add_argument("--adaptive-increase-factor", type=float, default=1.10,
                        help="APDP epsilon multiplier for above-average clients.")
    parser.add_argument("--adaptive-decrease-factor", type=float, default=0.90,
                        help="APDP epsilon multiplier for below-average clients.")
    parser.add_argument("--disable-compute-epoch-scaling", action="store_true",
                        help="Disable heterogeneous compute local-epoch scaling while keeping epsilon compensation.")
    parser.add_argument("--participation-rate", type=float, default=0.6,
                        help="Client participation ratio for --experiment-suite participation.")
    parser.add_argument("--participation-policies", default="all,random,apdp_score",
                        help="Participation policies: all,random,apdp_score.")
    parser.add_argument("--privacy-budgets", default="20,50,80,100",
                        help="Comma-separated total privacy budgets for privacy sensitivity experiments.")
    parser.add_argument("--privacy-sensitivity-methods", default="ldp_fl,global_dp,dp_rtfl,apdp_rtfl",
                        help="Methods for privacy sensitivity experiments.")
    parser.add_argument("--enable-fairness-evaluation", action="store_true",
                        help="Enable client-level fairness evaluation in sklearn experiment suites.")
    parser.add_argument("--fairness-methods", default="ldp_fl,global_dp,dp_rtfl,apdp_rtfl",
                        help="Methods for client-level fairness experiments.")
    parser.add_argument("--synthetic-sensitive-attrs", default="gender,age,region",
                        help="Synthetic client-level sensitive attributes for fairness pressure tests.")
    parser.add_argument("--fairness-pressure-profile", default="regulated", choices=["regulated"],
                        help="Synthetic fairness pressure profile.")
    parser.add_argument("--fairness-datasets", default="emnist",
                        help="Datasets intended for synthetic fairness pressure tests.")
    parser.add_argument("--enable-contribution-evaluation", action="store_true",
                        help="Enable penalty and approximate Shapley contribution evaluation in sklearn experiment suites.")
    parser.add_argument("--contribution-methods", default="ldp_fl,global_dp,dp_rtfl,apdp_rtfl",
                        help="Methods for penalty and Shapley contribution experiments.")
    parser.add_argument("--contribution-quality-weight", type=float, default=0.25,
                        help="Weight for data-quality score in the final contribution score.")
    parser.add_argument("--contribution-shapley-weight", type=float, default=0.35,
                        help="Weight for approximate Shapley score in the final contribution score.")
    parser.add_argument("--contribution-risk-weight", type=float, default=0.30,
                        help="Weight for regulatory risk penalty in the final contribution score.")
    parser.add_argument("--contribution-fairness-weight", type=float, default=0.10,
                        help="Weight for client fairness penalty in the final contribution score.")
    parser.add_argument("--contribution-utility-metric", default="balanced_accuracy",
                        choices=["accuracy", "balanced_accuracy", "f1_score"],
                        help="Utility metric used for leave-one-out approximate Shapley evaluation.")
    parser.add_argument("--audit-methods", default="apdp_rtfl",
                        help="Methods for audit trace experiments.")
    parser.add_argument("--audit-digest-algorithm", default="sha256", choices=["sha256"],
                        help="Digest algorithm for the audit trace hash chain.")
    parser.add_argument("--ablation-method", default="apdp_rtfl", choices=["apdp_rtfl", "dp_rtfl"],
                        help="Base method for ablation experiments.")
    parser.add_argument("--ablation-scenarios", default="full,no_adaptive_privacy,no_compute_adapter,no_zkip,no_ebcd,no_tcm,no_regulatory,no_contribution,no_fairness",
                        help="Comma-separated ablation scenarios.")
    parser.add_argument("--enable-regulatory-intervention", action="store_true",
                        help="Enable regulatory warning, downweighting, and quarantine in sklearn baseline suites.")
    parser.add_argument("--reg-warning-threshold", type=float, default=1.5,
                        help="Regulatory risk threshold for warning actions.")
    parser.add_argument("--reg-quarantine-threshold", type=float, default=2.5,
                        help="Regulatory risk threshold for quarantine actions.")
    parser.add_argument("--reg-penalty-weight", type=float, default=0.5,
                        help="Penalty weight used to downweight risky client updates.")
    parser.add_argument("--enable-pollution-injection", action="store_true",
                        help="Enable data pollution injection inside sklearn experiment suites.")
    parser.add_argument("--pollution-type", default="label_flip", choices=["label_flip", "feature_noise"],
                        help="Data pollution type for pollution experiments.")
    parser.add_argument("--polluted-clients", default="1",
                        help="Comma-separated zero-based client indices to pollute, e.g. 1,3.")
    parser.add_argument("--pollution-start-round", type=int, default=1,
                        help="First round to inject pollution.")
    parser.add_argument("--pollution-end-round", type=int, default=0,
                        help="Last round to inject pollution. Use 0 to continue through the final round.")
    parser.add_argument("--pollution-rate", type=float, default=0.3,
                        help="Fraction of a polluted client's local samples to corrupt per polluted round.")
    parser.add_argument("--pollution-feature-noise-std", type=float, default=2.0,
                        help="Gaussian feature noise standard deviation for feature_noise pollution.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def create_output_dir(args):
    run_name = args.run_name or f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_root = args.output_root if os.path.isabs(args.output_root) else os.path.join(repo_root, args.output_root)
    output_dir = os.path.join(output_root, run_name)
    os.makedirs(output_root, exist_ok=True)
    if os.path.exists(output_dir):
        raise FileExistsError(
            f"Run directory already exists: {output_dir}. Choose a new --run-name to avoid mixing experiment artifacts."
        )
    os.makedirs(output_dir)
    return output_dir

# 隐私预算分配器
class PrivacyBudgetAllocator:
    def __init__(self, total_budget, strategy="hybrid", warmup_rounds=20,
                 increase_factor=1.10, decrease_factor=0.90):
        self.total_budget = total_budget
        self.strategy = strategy
        self.warmup_rounds = warmup_rounds
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.clients_statistics = {} # 存储客户端统计信息；历史记录
        self.last_round_contributions = {} # 上一轮动态贡献
    # 计算数据质量分数（基于类别分布的熵）
    def compute_data_quality_score(self, y_data):
        if y_data is None or len(y_data) == 0:
            return 0.0  
        # 计算类别分布
        unique, counts = np.unique(y_data, return_counts=True)
        probabilities = counts / len(y_data)
        # 计算熵（熵值越高表示数据分布越均匀，质量越好）
        data_entropy = entropy(probabilities)
        max_entropy = np.log(len(unique)) if len(unique) > 0 else 1.0
        normalized_entropy = data_entropy / max_entropy if max_entropy > 0 else 0.0
        return normalized_entropy
    # 为活跃客户分配隐私预算
    def allocate_budget(self, clients, active_client_indices):
        if not active_client_indices:
            return []
        n_active = len (active_client_indices)
        # 均匀分配
        if self.strategy == "uniform":
            epsilon_per_client = self.total_budget / n_active
            return {client_id: epsilon_per_client for client_id in active_client_indices}
        # 基于数据量分配
        elif self.strategy == "data-size":
            total_data = sum(len(clients[i]. y_train) for i in active_client_indices)
            if total_data == 0:
                epsilon_per_client = self.total_budget / n_active
                return {client_id: epsilon_per_client for client_id in active_client_indices}
            allocations = {}
            for i in active_client_indices:
                data_weight = len(clients[i]. y_train) / total_data
                allocations[i] = self.total_budget * data_weight
            return allocations
        #基于数据质量分配
        elif self.strategy == "data_quality":
            quality_scores = {}
            total_quality = 0.0
            for i in active_client_indices:
                quality = self.compute_data_quality_score(clients[i]. y_train)
                quality_scores[i] = quality
                total_quality += quality
            if total_quality == 0:
                epsilon_per_client = self.total_budget / n_active
                return {client_id: epsilon_per_client for client_id in active_client_indices}
            allocations = {}
            for i in active_client_indices:
                quality_weight = quality_scores[i] / total_quality
                allocations[i] = self.total_budget * quality_weight
            return allocations
        # 混合分配：结合数据量和数据质量
        elif self.strategy == "hybrid":
            total_data = sum(len(clients[i].y_train) for i in active_client_indices)
            if total_data == 0:
                epsilon_per_client = self.total_budget / n_active
                return {client_id: epsilon_per_client for client_id in active_client_indices}

            # 新增：计算算力补偿因子（低算力→更高权重→更高隐私预算）
            total_compute_factor = 0.0
            compute_factors = {}
            for i in active_client_indices:
                cap = getattr(clients[i], 'compute_capability', 1.0)
                # 低算力→获得更高权重（补偿→更高隐私预算）
                compute_factor = 1.0 / (cap + 0.01)
                compute_factors[i] = compute_factor
                total_compute_factor += compute_factor

            allocations = {}
            total_weight = 0.0
            for i in active_client_indices:
                # 数据量权重（70%）
                data_weight = len(clients[i].y_train) / total_data
                # 数据质量权重（30%）
                quality = self.compute_data_quality_score(clients[i].y_train)
                # 定义compute_weight
                compute_weight = (
                    compute_factors[i] / total_compute_factor
                    if total_compute_factor > 0 else 0.0
                )
                # 组合权重
                weight = 0.50 * data_weight + 0.25 * quality + 0.25 * compute_weight
                allocations[i] = weight
                total_weight += weight
            # 归一化并分配预算
            if total_weight > 0:
                for i in allocations:
                    allocations[i] = (allocations[i] / total_weight) * self.total_budget
            else:
                epsilon_per_client = self.total_budget / n_active
                for i in active_client_indices:
                    allocations[i] = epsilon_per_client
            return allocations
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

    # 每轮训练结束后，更新上一轮贡献记录
    def update_contributions(self, round_num, client_contributions):
        self.last_round_contributions = client_contributions.copy()

    # 基于历史表现自适应调整隐私预算
    def adaptive_adjustment(self, clients, previous_allocations, round_num):
        if round_num <= self.warmup_rounds:
            return previous_allocations.copy()
        # 简单的调整策略： 基于客户端的历史贡献
        new_allocations = previous_allocations.copy()
        contributions = self.last_round_contributions # 使用上一轮贡献
        if not hasattr(self, 'clients_statistics') or not isinstance(self.clients_statistics, dict):
            self.clients_statistics = {}
        for i, epsilon in previous_allocations.items():
            if i not in self.clients_statistics:
                self.clients_statistics[i] = {
                    'contribution_history': [],
                    'data_quality_history': [],
                    'epsilon_history': []
                }
            # 获取上一轮贡献（优先使用复合指标）
            contrib = contributions.get(i, 0.0)
            if isinstance(contrib, (list, tuple)) and len(contrib) >=3:
                update_norm, acc_gain, loss_drop = contrib
                # 综合贡献分数（可调整权重）
                contribution_score = 0.5 * update_norm + 0.3 * acc_gain + 0.2 * loss_drop
            else:
                contribution_score = float(contrib) if contrib else 0.0
            # 计算客户端贡献（基于数据量和质量）
            data_size = len(clients[i].y_train) if hasattr(clients[i], 'y_train') else 0
            data_quality = self.compute_data_quality_score(clients[i].y_train) if hasattr(clients[i],
                                                                                          'y_train') else 0.5
            contribution_score = data_size * data_quality
            # 更新统计信息
            self.clients_statistics[i]['contribution_history'].append(contribution_score)
            self.clients_statistics[i]['data_quality_history'].append(data_quality)
            self.clients_statistics[i]['epsilon_history'].append(epsilon)
            # 基于历史表现调整
            if len(self.clients_statistics[i]['contribution_history']) > 1:
                recent_avg = np.mean(self.clients_statistics[i]['contribution_history'][-3:])
                all_recent = [np.mean(s['contribution_history'][-3:])
                              for s in self.clients_statistics.values()
                              if len(s['contribution_history']) >= 3]
                global_avg = np.mean(all_recent) if all_recent else recent_avg
                if recent_avg > global_avg * 1.05: # 贡献显著高于平均
                    new_allocations[i] = min(MAX_EPSILON, epsilon * self.increase_factor)
                else: # 贡献显著低于平均
                    new_allocations[i] = max(MIN_EPSILON, epsilon * self.decrease_factor)
        # 确保总预算不变
        total_allocated = sum(new_allocations.values())
        if total_allocated > 0:
            scaling = self.total_budget / total_allocated
            for i in new_allocations:
                new_allocations[i] = max(MIN_EPSILON, min(MAX_EPSILON,new_allocations[i] * scaling))
        return new_allocations

class HeterogeneousComputeAdapter:
    """异构算力适配模块：处理设备计算能力差异"""
    def __init__(self, capabilities=None):
        # capabilities: dict {client_idx: relative_speed (0.2~1.0)}
        self.capabilities = capabilities or {}
        self.min_cap_threshold = 0.3  # 低于此阈值视为极低算力

    def assign_capabilities(self, num_clients):
        # 模拟真实跨设备分布：高、中、低算力
        default_dist = [1.0, 0.85, 0.65, 0.40, 0.25][:num_clients]
        if len(default_dist) < num_clients:
            default_dist += [0.5] * (num_clients - len(default_dist))
        self.capabilities = {i: default_dist[i] for i in range(num_clients)}
        return self.capabilities

    def get_effective_epochs(self, client_idx, base_epochs):
        """根据算力动态降低本地训练轮次（主计算开销来源）"""
        if getattr(self, "disable_epoch_scaling", False):
            return base_epochs
        cap = self.capabilities.get(client_idx, 1.0)
        effective = max(1, int(base_epochs * cap * 1.1))  # 轻微上浮避免过低
        return effective

    def adjust_epsilon_for_compute(self, base_epsilon, client_idx):
        """低算力设备 → 更高epsilon（更弱隐私、更小噪声量，降低加噪开销）"""
        cap = self.capabilities.get(client_idx, 1.0)
        if cap < self.min_cap_threshold:
            return MAX_EPSILON  # 极低算力几乎不加噪声
        # 线性映射：算力越低 → epsilon越高
        adjustment = 1.0 + (1.0 - cap) * 0.8
        new_eps = base_epsilon * adjustment
        return min(MAX_EPSILON, max(MIN_EPSILON, new_eps))

def main():
    args = parse_args()
    np.random.seed(args.seed)
    output_dir = create_output_dir(args)
    initialize_run_artifacts(output_dir, args)
    global TOTAL_PRIVACY_BUDGET, MIN_EPSILON, MAX_EPSILON, DP_EPSILON, DP_DELTA, DP_L2_NORM_CLIP
    TOTAL_PRIVACY_BUDGET = args.total_privacy_budget
    MIN_EPSILON = args.min_epsilon
    MAX_EPSILON = args.max_epsilon
    DP_EPSILON = args.dp_epsilon
    DP_DELTA = args.dp_delta
    DP_L2_NORM_CLIP = args.dp_l2_norm_clip
    if args.experiment_suite in {"baselines", "participation", "privacy_sensitivity", "pollution", "fairness", "synthetic_fairness", "contribution", "audit_trace", "ablation"}:
        if args.backend == "torch":
            from torch_baselines import run_torch_baseline_suite
            if args.experiment_suite != "baselines":
                raise NotImplementedError("Torch backend currently supports --experiment-suite baselines. Use --backend sklearn for participation/privacy_sensitivity/pollution/fairness/synthetic_fairness/contribution/audit_trace/ablation suites.")
            run_torch_baseline_suite(args, output_dir)
        else:
            from baselines import run_baseline_suite, run_participation_suite, run_privacy_sensitivity_suite, run_pollution_injection_suite, run_fairness_suite, run_synthetic_fairness_suite, run_contribution_suite, run_audit_trace_suite, run_ablation_suite
            if args.experiment_suite == "baselines":
                run_baseline_suite(args, output_dir)
            elif args.experiment_suite == "participation":
                run_participation_suite(args, output_dir)
            elif args.experiment_suite == "privacy_sensitivity":
                run_privacy_sensitivity_suite(args, output_dir)
            elif args.experiment_suite == "pollution":
                run_pollution_injection_suite(args, output_dir)
            elif args.experiment_suite == "fairness":
                run_fairness_suite(args, output_dir)
            elif args.experiment_suite == "synthetic_fairness":
                run_synthetic_fairness_suite(args, output_dir)
            elif args.experiment_suite == "contribution":
                run_contribution_suite(args, output_dir)
            elif args.experiment_suite == "audit_trace":
                run_audit_trace_suite(args, output_dir)
            else:
                run_ablation_suite(args, output_dir)
        return
    if args.backend == "torch":
        from torch_baselines import run_torch_baseline_suite
        args.methods = "apdp_rtfl" if args.methods == "all" else args.methods
        run_torch_baseline_suite(args, output_dir)
        return
    print("Starting RTFL Simulation with Differential Privacy...")
    print(f"Dataset: {args.dataset}")
    print(f"Data root: {args.data_root}")
    print(f"Results will be saved to: {output_dir}")
    print(f"Total Privacy Budget: {TOTAL_PRIVACY_BUDGET}")
    print(f"Allocation Strategy: {PRIVACY_ALLOCATION_STRATEGY}")
    print(f"DP Parameters: Epsilon={DP_EPSILON}, Delta={DP_DELTA}, L2_Norm_Clip={DP_L2_NORM_CLIP}")
    print(f"Epsilon bounds: min={MIN_EPSILON}, max={MAX_EPSILON}. Failure probability={args.failure_prob}")
    print(
        "APDP tuning: "
        f"warmup_rounds={args.apdp_warmup_rounds}, "
        f"increase_factor={args.adaptive_increase_factor}, "
        f"decrease_factor={args.adaptive_decrease_factor}, "
        f"disable_compute_epoch_scaling={args.disable_compute_epoch_scaling}"
    )
    # 初始化隐私预算分配器
    privacy_allocator = PrivacyBudgetAllocator(
        TOTAL_PRIVACY_BUDGET,
        PRIVACY_ALLOCATION_STRATEGY,
        warmup_rounds=args.apdp_warmup_rounds,
        increase_factor=args.adaptive_increase_factor,
        decrease_factor=args.adaptive_decrease_factor,
    )

    X_train_full, y_train_full, X_test, y_test, feature_names, classes, presplit_client_data = load_experiment_data(
        dataset_name=args.dataset,
        data_root=args.data_root,
        random_state=args.seed,
        max_samples=args.max_samples,
        emnist_split=args.emnist_split,
    )
    if X_train_full is None:
        print("Failed to load data. Exiting.")
        return
    num_features = X_train_full.shape[1]
    print(f"Data loaded: {X_train_full.shape[0]} train samples, {X_test.shape[0]} test samples. Num features: {num_features}, Classes: {classes}")
    if presplit_client_data is not None:
        client_datasets = presplit_client_data
        args.num_clients = len(client_datasets)
        print(f"Using {args.num_clients} pre-split clients from {args.dataset}/all_data.")
    else:
        client_datasets = split_data_for_clients(X_train_full,
            y_train_full,
            args.num_clients,
            size_ratios=None,
            partition=args.partition,
            dirichlet_alpha=args.dirichlet_alpha,
            random_state=args.seed)
    clients = []
    client_ids = [f"client_{i}" for i in range(args.num_clients)]
    # 显示客户端数据分布信息
    print("\nClient Data Distribution:")
    for i in range(args.num_clients):
        X_c, y_c = client_datasets[i]
        unique, counts = np.unique(y_c, return_counts=True) if len(y_c) > 0 else ([], [])
        distribution = dict(zip(unique, counts)) if len(unique) > 0 else {}
        data_quality = privacy_allocator.compute_data_quality_score(y_c)
        print(f"Client {i}: Data size={len(y_c)}, Classes={len(unique)}, Data quality={data_quality: .3f}")

    # Split each client's data into train/val
    client_val_sets = []
    for i in range(args.num_clients):
        X_c, y_c = client_datasets[i]
        if X_c.shape[0] > 5 and len(np.unique(y_c)) >= 2:
            stratify = y_c if min(np.bincount(y_c, minlength=len(classes))) >= 2 else None
            X_c_train, X_c_val, y_c_train, y_c_val = train_test_split(
                X_c, y_c, test_size=0.2, random_state=args.seed + i, stratify=stratify
            )
        else:
            X_c_train, y_c_train = X_c, y_c
            X_c_val, y_c_val = None, None
        # 使用默认隐私预算初始化客户端，后续会动态调整
        clients.append(FLClient(client_ids[i], X_c_train, y_c_train, num_features, 
                                learning_rate=BASE_LEARNING_RATE,
                                dp_epsilon=DP_EPSILON, # 初始值，会被动态覆盖
                                dp_delta=DP_DELTA, 
                                dp_l2_norm_clip=DP_L2_NORM_CLIP,
                                random_state=i,
                                X_val=X_c_val, y_val=y_c_val, earlystop_patience=EARLYSTOP_PATIENCE,
                                classes=classes))
        client_val_sets.append((X_c_val, y_c_val))
    main_failure_rng = np.random.default_rng(args.seed + 2026)
    failure_plan = main_failure_rng.random((args.num_rounds, args.num_clients)) < args.failure_prob
    train_val_data = [
        (client.X_train, client.y_train, client.X_val, client.y_val)
        for client in clients
    ]
    write_data_artifacts(output_dir, args, train_val_data, X_test, y_test, classes, failure_plan)
    # 新增：异构算力适配器
    compute_adapter = HeterogeneousComputeAdapter()
    compute_adapter.disable_epoch_scaling = args.disable_compute_epoch_scaling
    # 为所有客户端分配计算能力（可以自定义，也可以随机模拟）
    capabilities = compute_adapter.assign_capabilities(args.num_clients)
    # 将计算能力绑定到每个客户端对象上
    print("\n=== Client Compute Capabilities ===")
    for i, client in enumerate(clients):
        client.compute_capability = capabilities.get(i,1.0)
        data_size = len(client.y_train) if hasattr(client, 'y_train') else 0
        print(f"Client {i:2d} ({client.client_id}):"
              f"data_size = {data_size:4d},"
              f"compute_capability = {client.compute_capability:3f}")
    print(f"Compute capability distribution: {list(capabilities.values())}")
    print("=======================================\n")


    # For server validation, concatenate all client val sets
    X_val_server = np.concatenate([v[0] for v in client_val_sets if v[0] is not None]) if any(v[0] is not None for v in client_val_sets) else None
    y_val_server = np.concatenate([v[1] for v in client_val_sets if v[1] is not None]) if any(v[1] is not None for v in client_val_sets) else None
    server_id = "main_server"
    server = FLServer(server_id, client_ids, num_features, X_val=X_val_server, y_val=y_val_server,
                      earlystop_patience=EARLYSTOP_PATIENCE, classes=classes)
    initial_params_for_ebcd = [client.model_parameters() for client in clients if client.X_train.shape[0] > 0]
    if initial_params_for_ebcd:
        server.ebcd.establish_baseline(initial_params_for_ebcd)

    # --- Metrics storage ---
    rounds = []
    accuracies = []
    f1_scores = []
    aucs = []
    ebcd_variances = []
    ebcd_kurtoses = []
    ebcd_skewnesses = []
    server_statuses = []
    coordinator_ids = []
    dp_noise_scales = []
    agg_client_counts = []
    zkip_failures = []
    delta_norms = []
    ebcd_alerts = []
    tcm_counts = []

    # 隐私预算相关指标
    privacy_budget_allocations = [] # 每轮的隐私预算分配
    client_data_qualities = [] # 客户端数据质量
    client_contribution_scores = [] # 客户端贡献分数

    # Per-client metrics: [round][client]
    per_client_update_norms = []
    per_client_ebcd_stats = []
    per_client_zkip_status = []
    per_client_epsilon = []

    # 第一轮隐私预算分配
    active_client_indices = list(range(len(clients)))
    initial_epsilon_allocations = privacy_allocator.allocate_budget(clients, active_client_indices)
    current_epsilon_allocations = initial_epsilon_allocations.copy()

    for round_num in range(1, args.num_rounds + 1):
        print(f"\n--- Round {round_num}/{args.num_rounds} ---")
        start_time = time.time()
        # 动态调整隐私预算（从第二轮开始）
        if round_num > 1:
            current_epsilon_allocations = privacy_allocator.adaptive_adjustment(clients, current_epsilon_allocations,
                                                                                round_num)
        print(f"Privacy Budget Allocation: {current_epsilon_allocations}")

        global_params_for_round = server.get_global_model_parameters_for_clients()
        if global_params_for_round is None:
            print(f"Round {round_num}: Critical - Could not get model from coordinator. Attempting TCM recovery.")
            latest_tcm_entry = server.tcm.get_latest_state_info()
            if latest_tcm_entry:
                recovered_params, _ = server.tcm.recover_state_by_round(latest_tcm_entry[1]) 
                if recovered_params:
                    server.global_model_parameters = recovered_params
                    global_params_for_round = server.get_global_model_parameters_for_clients() 
                    print(f"Round {round_num}: Recovered model from TCM (Round {latest_tcm_entry[1]}).")
                else:
                    print(f"Round {round_num}: TCM recovery failed. Skipping round.")
                    continue
            else:
                 print(f"Round {round_num}: No TCM entries to recover from. Skipping round.")
                 continue
        client_deltas_with_proofs = []
        client_data_sizes_for_agg = []
        active_clients_this_round_ids = []
        # --- DP noise scale for this round (all clients, average) ---
        round_noise_scales = []
        round_zkip_failures = 0
        round_delta_norm = 0.0
        round_ebcd_alert = 0
        round_client_update_norms = []
        round_client_ebcd_stats = []
        round_client_zkip_status = []
        round_client_epsilon_list = [] # 本轮每个客户端的隐私预算
        for idx, client in enumerate(clients):
            if failure_plan[round_num - 1][idx]:
                round_client_update_norms.append(None)
                round_client_ebcd_stats.append((None, None, None))
                round_client_zkip_status.append(None)
                continue 
            active_clients_this_round_ids.append(client.client_id)

            # 设置动态隐私预算
            client_epsilon  = current_epsilon_allocations.get(idx, DP_EPSILON)
            # 新增：根据算力进一步调整epsilon和epochs
            adjusted_epsilon = compute_adapter.adjust_epsilon_for_compute(client_epsilon, idx)
            effective_epochs = compute_adapter.get_effective_epochs(idx, args.client_epochs)

            client.dp_epsilon = max(MIN_EPSILON, min(MAX_EPSILON, adjusted_epsilon))
            round_client_epsilon_list.append(client.dp_epsilon)

            # 打印观察效果
            print(f"Client {idx} ({client.client_id}):"
                  f"cap={getattr(client, 'compute_capability', 1.0):.2f},"
                  f"effective_epochs={effective_epochs}, epsilon={client.dp_epsilon:.3f}")

            client.set_global_model_parameters(global_params_for_round)
            # 使用动态的本地训练轮次
            delta_weights, proof = client.train(epochs=effective_epochs)
            # ZKIP proof check (simulate server-side check)
            if delta_weights is not None and proof is not None:
                from zkip import ZeroKnowledgeIntegrityProofs
                zkip = ZeroKnowledgeIntegrityProofs()
                zkip_status = zkip.verify_proof(delta_weights, proof)
                round_client_zkip_status.append(zkip_status)
                # Delta norm (L2)
                norm = 0.0
                for v in delta_weights.values():
                    norm += np.linalg.norm(v.flatten())**2
                update_norm = np.sqrt(norm)
                round_client_update_norms.append(update_norm)
                # Per-client EBCD stats (variance, kurtosis, skewness) for delta_weights['coef_']
                if 'coef_' in delta_weights and hasattr(delta_weights['coef_'], 'flatten'):
                    flat = delta_weights['coef_'].flatten()
                    v = np.var(flat)
                    k = kurtosis(flat, fisher=True)
                    s = skew(flat)
                    round_client_ebcd_stats.append((v, k, s))
                else:
                    round_client_ebcd_stats.append((None, None, None))
                if not zkip_status:
                    round_zkip_failures += 1
                round_delta_norm += update_norm
                client_deltas_with_proofs.append((delta_weights, proof, client.client_id))
                client_data_sizes_for_agg.append(len(client.y_train))
                # DP noise scale: (client.dp_l2_norm_clip * np.sqrt(2 * np.log(1.25 / client.dp_delta))) / client.dp_epsilon
                if client.dp_epsilon > 0:
                    noise_stddev = (client.dp_l2_norm_clip * np.sqrt(2 * np.log(1.25 / client.dp_delta))) / client.dp_epsilon
                else:
                    noise_stddev = 0.0
                round_noise_scales.append(noise_stddev)
            else:
                round_client_update_norms.append(None)
                round_client_ebcd_stats.append((None, None, None))
                round_client_zkip_status.append(False)
        # Average delta norm for the round
        if len([n for n in round_client_update_norms if n is not None]) > 0:
            round_delta_norm = round_delta_norm / len([n for n in round_client_update_norms if n is not None])
        else:
            round_delta_norm = 0.0
        per_client_update_norms.append(round_client_update_norms)
        per_client_ebcd_stats.append(round_client_ebcd_stats)
        per_client_zkip_status.append(round_client_zkip_status)
        per_client_epsilon.append(round_client_epsilon_list) # 记录本轮隐私预算分配
        server.arrp.update_active_clients(active_clients_this_round_ids)
        aggregation_success, aggregated_from_clients = server.aggregate_model_deltas(client_deltas_with_proofs, client_data_sizes_for_agg)
        # EBCD alert (after aggregation)
        ebcd_alert = 1 if server.ebcd.check_for_corruption(server.global_model_parameters) else 0
        round_ebcd_alert = ebcd_alert
        server_state_details = {
            'arrp_status': server.arrp.status.name,
            'current_coordinator': server.arrp.get_current_coordinator_id(),
            'aggregation_successful': aggregation_success,
            'aggregated_from_clients_count': len(aggregated_from_clients),
            'dp_epsilon': DP_EPSILON,
            'dp_l2_norm_clip': DP_L2_NORM_CLIP
        }
        client_updates_summary = {cid: "OK_DP" for cid in aggregated_from_clients} 
        for cid in active_clients_this_round_ids:
            if cid not in client_updates_summary: client_updates_summary[cid] = "NO_UPDATE_OR_FAULTY"
        server.tcm.record_state(round_num, server.global_model_parameters, server_state_details, client_updates_summary)
        metrics = server.evaluate_global_model(X_test, y_test, round_num)
        print(f"Round {round_num} Eval (DP): Acc={metrics.get('accuracy',0):.3f}, F1={metrics.get('f1_score',0):.3f}, AUC={metrics.get('auc_roc',0):.3f}")
        round_duration = time.time() - start_time
        print(f"Round {round_num} duration: {round_duration:.2f}s. Coordinator: {server.arrp.get_current_coordinator_id()}")
        # --- Metrics collection ---
        rounds.append(round_num)
        accuracies.append(metrics.get('accuracy', 0))
        f1_scores.append(metrics.get('f1_score', 0))
        aucs.append(metrics.get('auc_roc', 0))
        agg_client_counts.append(len(aggregated_from_clients))
        server_statuses.append(server.arrp.status.name)
        coordinator_ids.append(server.arrp.get_current_coordinator_id())
        dp_noise_scales.append(np.mean(round_noise_scales) if round_noise_scales else 0.0)
        zkip_failures.append(round_zkip_failures)
        delta_norms.append(round_delta_norm)
        ebcd_alerts.append(round_ebcd_alert)
        tcm_counts.append(len(server.tcm.manifold_log))

        # 计算本轮每个客户端的动态贡献指标
        client_contributions = {}
        for idx, client in enumerate(clients):
            update_norm = round_client_update_norms[idx] if idx < len(round_client_update_norms) and round_client_update_norms[idx] is not None else 0.0
            # 本地验证准确率提升（当前版本可能还没有记录，先用0.0占位）
            acc_gain = getattr(client, 'last_val_acc_gain', 0.0)
            # 验证随时下降幅度（loss_drop = before - after, 越大越好）
            loss_drop = getattr(client, 'last_val_loss_drop', 0.0)
            client_contributions[idx] = (update_norm, acc_gain, loss_drop)
        # 更新allocator 的上一轮贡献记录
        privacy_allocator.update_contributions(round_num, client_contributions)
        # 记录隐私预算分配
        privacy_budget_allocations.append(current_epsilon_allocations.copy())

        # EBCD stats (variance, kurtosis, skewness) for global model coef_
        coef = server.global_model_parameters['coef_']
        if coef is not None and hasattr(coef, 'flatten'):
            flat = coef.flatten()
            ebcd_variances.append(np.var(flat))
            ebcd_kurtoses.append(kurtosis(flat, fisher=True))
            ebcd_skewnesses.append(skew(flat))
        else:
            ebcd_variances.append(0)
            ebcd_kurtoses.append(0)
            ebcd_skewnesses.append(0)

    print("\n--- RTFL Simulation with DP Complete ---")
    print(f"Total states recorded by TCM: {len(server.tcm.manifold_log)}")
    print(f"Pravacy allocation strategy: {PRIVACY_ALLOCATION_STRATEGY}")
    print(f"Fianl privacy budget allocations: {current_epsilon_allocations}")

    if args.num_rounds >= 3:
        target_recovery_round = args.num_rounds // 2
        print(f"\nAttempting to recover model state from TCM for round {target_recovery_round}...")
        recovered_model_params, rec_info = server.tcm.recover_state_by_round(target_recovery_round)
        if recovered_model_params:
            print(f"Successfully recovered (noisy) model parameters for round {target_recovery_round}.")
        else:
            print(f"Failed to recover model for round {target_recovery_round}.")
    # --- Save per-client metrics as .npy for research ---
    np.save(os.path.join(output_dir, 'per_client_update_norms.npy'), np.array(per_client_update_norms, dtype=object))
    np.save(os.path.join(output_dir, 'per_client_ebcd_stats.npy'), np.array(per_client_ebcd_stats, dtype=object))
    np.save(os.path.join(output_dir, 'per_client_zkip_status.npy'), np.array(per_client_zkip_status, dtype=object))
    np.save(os.path.join(output_dir, 'per_client_epsilon.npy'), np.array(per_client_epsilon, dtype=object)) # 保存隐私预算分配
    single_metric_rows = [
        {
            "round": rounds[idx],
            "accuracy": accuracies[idx],
            "f1_score": f1_scores[idx],
            "auc_roc": aucs[idx],
            "dp_noise_scale": dp_noise_scales[idx],
            "agg_client_count": agg_client_counts[idx],
            "zkip_failures": zkip_failures[idx],
            "delta_norm": delta_norms[idx],
            "ebcd_alert": ebcd_alerts[idx],
            "tcm_state_count": tcm_counts[idx],
            "ebcd_variance": ebcd_variances[idx],
            "ebcd_kurtosis": ebcd_kurtoses[idx],
            "ebcd_skewness": ebcd_skewnesses[idx],
        }
        for idx in range(len(rounds))
    ]
    earlystop_best_metric = (
        server.earlystop.best_metric
        if hasattr(server, "earlystop") and server.earlystop.best_metric != -float("inf")
        else np.nan
    )
    earlystop_best_metric_series = [earlystop_best_metric for _ in rounds]
    import csv
    with open(os.path.join(output_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(single_metric_rows[0].keys()) if single_metric_rows else ["round"])
        writer.writeheader()
        writer.writerows(single_metric_rows)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(
            json_safe({
                "round_metrics": single_metric_rows,
                "server_statuses": server_statuses,
                "coordinator_ids": coordinator_ids,
                "privacy_budget_allocations": privacy_budget_allocations,
                "earlystop_best_metric": earlystop_best_metric_series,
                "per_client_data_files": {
                    "update_norms": "per_client_update_norms.npy",
                    "ebcd_stats": "per_client_ebcd_stats.npy",
                    "zkip_status": "per_client_zkip_status.npy",
                    "epsilon": "per_client_epsilon.npy",
                },
            }),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    export_tcm_checkpoints(output_dir, server.tcm)
    # --- Save plots for research ---
    charts.plot_global_metrics(rounds, accuracies, f1_scores, aucs)
    charts.save_figure(os.path.join(output_dir, 'global_metrics.png'))
    plt.close()
    charts.plot_ebcd_stats(rounds, ebcd_variances, ebcd_kurtoses, ebcd_skewnesses)
    charts.save_figure(os.path.join(output_dir, 'ebcd_stats.png'))
    plt.close()
    # For server status, encode status as int for plotting
    status_map = {s: i for i, s in enumerate(sorted(set(server_statuses)))}
    status_ints = [status_map[s] for s in server_statuses]
    charts.plot_server_status(rounds, status_ints, coordinator_ids)
    charts.save_figure(os.path.join(output_dir, 'server_status.png'))
    plt.close()
    charts.plot_dp_noise_scale(rounds, dp_noise_scales)
    charts.save_figure(os.path.join(output_dir, 'dp_noise_scale.png'))
    plt.close()
    charts.plot_agg_client_counts(rounds, agg_client_counts)
    charts.save_figure(os.path.join(output_dir, 'agg_client_counts.png'))
    plt.close()
    charts.plot_zkip_failures(rounds, zkip_failures)
    charts.save_figure(os.path.join(output_dir, 'zkip_failures.png'))
    plt.close()
    charts.plot_delta_norm(rounds, delta_norms)
    charts.save_figure(os.path.join(output_dir, 'delta_norm.png'))
    plt.close()
    charts.plot_ebcd_alerts(rounds, ebcd_alerts)
    charts.save_figure(os.path.join(output_dir, 'ebcd_alerts.png'))
    plt.close()
    charts.plot_tcm_state_count(rounds, tcm_counts)
    charts.save_figure(os.path.join(output_dir, 'tcm_state_count.png'))
    plt.close()

    # 隐私预算分配可视化
    def plot_privacy_budget_allocation(rounds, privacy_allocations, client_ids): #绘制隐私预算分配图
        plt.figure(figsize=charts.FIGSIZE_WIDE)
        # 准备数据
        n_clients = len(client_ids)
        client_epsilon_history = [[] for _ in range(n_clients)]
        for round_alloc in privacy_allocations:
            for i in range(n_clients):
                epsilon = round_alloc.get(i,0) if i in round_alloc else 0
                client_epsilon_history[i].append(epsilon)
        # 绘制每个客户端的隐私预算变化
        for i in range(n_clients):
            plt.plot(rounds, client_epsilon_history[i], marker='o', label=f'Client {i}', linewidth=2)

        plt.xlabel('Round')
        plt.ylabel('Privacy Budget (ε)')
        plt.title(f'Dynamic Privacy Budget Allocation per Round\n'
                  f'Strategy:{PRIVACY_ALLOCATION_STRATEGY}, TOTAL_PRIVACY_BUDGET:{TOTAL_PRIVACY_BUDGET}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        charts.save_figure(os.path.join(output_dir, 'privacy_budget_allocation.png'))
        plt.close()

    #调用隐私预算分配可视化函数
    plot_privacy_budget_allocation(rounds, privacy_budget_allocations, client_ids)
    # 隐私预算与数据质量关系图
    def plot_epsilon_vs_data_quality(clients, final_allocations): # 绘制隐私预算与数据质量的关系图
        data_qualities = []
        epsilon_values = []
        data_sizes = []
        quality_rows = []
        for i, client in enumerate(clients):
            if hasattr(client, 'y_train'):
                quality = privacy_allocator.compute_data_quality_score(client.y_train)
                epsilon = final_allocations.get(i,0)
                data_qualities.append(quality)
                epsilon_values.append(epsilon)
                data_sizes.append(len(client.y_train))
                quality_rows.append(
                    {
                        "client_id": client.client_id,
                        "data_quality_entropy": quality,
                        "final_epsilon": epsilon,
                        "train_samples": len(client.y_train),
                    }
                )

        with open(os.path.join(output_dir, "client_privacy_quality_summary.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(quality_rows[0].keys()) if quality_rows else ["client_id"])
            writer.writeheader()
            writer.writerows(quality_rows)

        plt.figure(figsize=charts.FIGSIZE_DEFAULT)
        scatter = plt.scatter(data_qualities, epsilon_values, s=[s/10 for s in data_sizes],
                              alpha=0.7, c=data_sizes, cmap='viridis')
        plt.colorbar(scatter, label='Data Size')
        plt.xlabel('Data Quality (Entropy)')
        plt.ylabel('Final Privacy Budget (ε)')
        plt.title('Relationship between Data Quality and Privacy Budget Allocation')
        plt.grid(True, alpha=0.3)

        # 添加趋势线
        if len(data_qualities) > 1:
            z = np.polyfit(data_qualities, epsilon_values, 1)
            p = np.poly1d(z)
            plt.plot(data_qualities, p(data_qualities), '--', color='red', alpha=0.8, label='Trend line')
            plt.legend()

        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, 'epsilon_vs_data_quality.png'))
        plt.close()

    plot_epsilon_vs_data_quality(clients, current_epsilon_allocations)

    # --- Early stopping chart for server ---
    if hasattr(server, 'earlystop') and hasattr(server.earlystop, 'best_metric'):
        best_accs = earlystop_best_metric_series
        charts.plot_early_stopping_metric(rounds, best_accs, metric_name="Best Validation Accuracy", ylabel="Accuracy")
        charts.save_figure(os.path.join(output_dir, 'earlystop_server_best_val_acc.png'))
        plt.close()
    # --- Per-client plots ---
    charts.plot_per_client_update_norms(rounds, per_client_update_norms, client_ids)
    charts.save_figure(os.path.join(output_dir, 'per_client_update_norms.png'))
    plt.close()
    # Per-client EBCD stats: variance, kurtosis, skewness
    for stat_idx, stat_name in enumerate(['Variance', 'Kurtosis', 'Skewness']):
        # Build [client][round] shape for plotting
        stat_data = [[None for _ in range(len(rounds))] for _ in range(len(client_ids))]
        for r in range(len(rounds)):
            for c in range(len(client_ids)):
                try:
                    val = per_client_ebcd_stats[r][c][stat_idx] if per_client_ebcd_stats[r][c] is not None else None
                except (IndexError, TypeError):
                    val = None
                stat_data[c][r] = val
        charts.plot_per_client_ebcd_stats(rounds, stat_data, client_ids, stat_name)
        charts.save_figure(os.path.join(output_dir, f'per_client_ebcd_{stat_name.lower()}.png'))
        plt.close()
    charts.plot_per_client_zkip_status(rounds, per_client_zkip_status, client_ids)
    charts.save_figure(os.path.join(output_dir, 'per_client_zkip_status.png'))
    plt.close()

    # 每轮每个客户端的隐私预算可视化
    def plot_per_client_epsilon(rounds, per_client_epsilon, client_ids): # 绘制每个客户端的隐私预算变化
        plt.figure(figsize=charts.FIGSIZE_WIDE)
        for client_idx in range(len(client_ids)):
            epsilon_history = []
            for round_idx in range(len(rounds)):
                if (round_idx < len(per_client_epsilon) and
                client_idx < len(per_client_epsilon[round_idx]) and
                per_client_epsilon[round_idx][client_idx] is not None):
                    epsilon_history.append(per_client_epsilon[round_idx][client_idx])
                else:
                    epsilon_history.append(np.nan)

            plt.plot(rounds, epsilon_history, marker='o', label=f'Client {client_idx}, linewidth=2')

        plt.xlabel('Round')
        plt.ylabel('Privacy Budget (ε)')
        plt.title('Per_Client Privacy Budget per Round')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        charts.save_figure(os.path.join(output_dir, 'per_client_epsilon_dynamic_dp.png'))
        plt.close()

    plot_per_client_epsilon(rounds, per_client_epsilon, client_ids)
    write_artifact_manifest(output_dir)
    print(f"Experiment artifacts saved to: {output_dir}")

if __name__ == "__main__":
    main()

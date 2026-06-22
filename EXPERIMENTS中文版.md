# APDP-RTFL 实验命令手册（中文版）

本手册用于组织面向受监管行业的 APDP-RTFL 可复现实验。请在项目根目录执行全部命令：

```powershell
python APDP-RTFL/main.py <arguments>
```

默认结果路径为 `results/<run-name>/`。每次正式实验均应显式指定 `--run-name`。同一张比较表或同一幅图中的所有方法，必须使用相同的数据集划分、客户端数量、训练轮数、数据划分方式、隐私预算参数、随机种子集合和后端。

每个运行目录均采用一次写入规则：若 `--run-name` 已存在，程序会拒绝运行，以防止不同实验产物混入同一目录。每次新运行都会生成 `run_config.json`、`run_command.txt`、`environment.json`、`data_artifacts/` 和 `artifact_manifest.csv`。其中 `data_artifacts/` 保存数据集指纹、客户端划分摘要和已生成的掉线计划；启用 TCM 的方法目录还会保存 `tcm_manifest.csv` 与可恢复的 `checkpoints/*.npz` 文件。

## 通用正式实验设置

以下参数是 EMNIST 正式实验的起始设置。该配置采用 non-IID 划分和固定随机种子。每一条正式命令至少应使用三个随机种子重复运行，例如 `42`、`43`、`44`，并报告均值和标准差。

```powershell
--dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --min-epsilon 0.1 --max-epsilon 2 --dp-epsilon 1 --dp-delta 1e-5 --dp-l2-norm-clip 1 --backend sklearn --seed 42
```

仅用于功能验证的小样本测试可附加 `--max-samples 500 --num-rounds 2 --num-clients 5`。小样本结果不得写入论文。

## 1. DP 基线对照

默认的 `--methods all` 仅包含 DP 对照组：

`DP-FL`、`DP-FLProx`、`DP-FedSGD`、`Global-DP`、`DP-RTFL` 和 `APDP-RTFL`。

```powershell
python APDP-RTFL/main.py --experiment-suite baselines --methods all --run-name baseline_emnist_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --min-epsilon 0.1 --max-epsilon 2 --dp-epsilon 1 --dp-delta 1e-5 --dp-l2-norm-clip 1 --backend sklearn --seed 42
```

主要输出为 `baseline_final_metrics.csv`、`baseline_summary.csv`、`baseline_comparison.png` 和 `baseline_method_metadata.csv`。其中元数据文件保存了论文比较表对应的代码侧配置记录：

| 方法 | 项目内配置 | 参考文献 |
| --- | --- | --- |
| DP-FL | 客户端侧 DP 更新 | Arachchige et al., *Local differential privacy for deep learning*, IEEE IoT Journal, 2019/2020. |
| DP-FLProx | 客户端侧 DP 更新加 FedProx 近端项 | Li et al., *Federated optimization in heterogeneous networks*, MLSys, 2020. |
| DP-FedSGD | 客户端侧 DP 更新，强制一个本地 epoch | Auddy et al., *Statistical Limits and Efficient Algorithms for Differentially Private Federated Learning*, arXiv:2605.18656, 2026. |
| Global-DP | 聚合后服务器侧加噪 | 项目实现基线。 |
| DP-RTFL | DP 加 ZKIP、EBCD 和 TCM | 项目实现基线。 |
| APDP-RTFL | DP-RTFL 加自适应隐私与计算适配 | 本文所提方法。 |

`DP-FedSGD` 是本项目中用于受控对照的实现配置，不应表述为对所引论文的严格复现。`FedAvg`、`FedProx` 和 `LDP-FL` 仅可通过显式指定方法名运行，不应放入论文的主 DP 对照表。

## 2. 监管干预实验

该实验用于量化预警、降权、隔离及其对模型效用的影响。除监管开关外，其他参数应与 DP 基线命令完全一致。

```powershell
python APDP-RTFL/main.py --experiment-suite baselines --methods apdp_rtfl --enable-regulatory-intervention --reg-warning-threshold 1.5 --reg-quarantine-threshold 2.5 --reg-penalty-weight 0.5 --run-name regulatory_emnist_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --backend sklearn --seed 42
```

应将该结果与去掉 `--enable-regulatory-intervention` 的同一命令进行对照。主要输出包括 `regulatory_intervention_summary.csv`、`regulatory_actions.png` 和 `regulatory_risk_by_client.png`。

## 3. 数据污染识别与监管干预

普通训练不会启用污染。只有同时使用 `--experiment-suite pollution` 与 `--enable-pollution-injection` 时，污染注入才会生效。

标签翻转场景：

```powershell
python APDP-RTFL/main.py --experiment-suite pollution --methods apdp_rtfl --enable-pollution-injection --pollution-type label_flip --polluted-clients 1,3 --pollution-start-round 10 --enable-regulatory-intervention --run-name pollution_label_flip_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --backend sklearn --seed 42
```

特征噪声场景：

```powershell
python APDP-RTFL/main.py --experiment-suite pollution --methods apdp_rtfl --enable-pollution-injection --pollution-type feature_noise --polluted-clients 1,3 --pollution-start-round 10 --enable-regulatory-intervention --run-name pollution_feature_noise_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --backend sklearn --seed 42
```

请从 `pollution_final_metrics.csv`、`pollution_summary.csv`、`pollution_injection_summary.csv` 和 `pollution_detection_rate.png` 中报告效用变化、识别率、假阳性率和假阴性率。

## 4. 合成敏感属性公平性压力测试

该实验属于客户端级的合成压力测试，而非对真实人口统计公平性的验证。它构造样本覆盖、标签分布、特征质量、参与稳定性和计算能力之间存在关联的群体差异。

```powershell
python APDP-RTFL/main.py --experiment-suite synthetic_fairness --fairness-datasets emnist --fairness-methods dp_fl,dp_flprox,dp_fedsgd,global_dp,dp_rtfl,apdp_rtfl --synthetic-sensitive-attrs gender,age,region --fairness-pressure-profile regulated --run-name synthetic_fairness_emnist_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --backend sklearn --seed 42
```

对于 FEMNIST、CIFAR10 和 CIFAR100，应先生成 `data/<dataset>/all_data`，再将 `--fairness-datasets emnist` 替换为所需数据集的逗号分隔列表。主要输出包括 `synthetic_sensitive_clients.csv`、`synthetic_group_fairness_summary.csv`、`federated_group_fairness_summary.csv` 及群体公平性图表。

论文建议表述：本文使用客户端级合成敏感属性模拟机构异质性与资源不均衡，该设置不代表真实的性别、年龄或地区人口统计属性。

## 5. 惩罚机制与近似 Shapley 贡献评估

当前贡献评估器以留一法边际效用计算近似 Shapley 值。论文中应明确称其为“近似 Shapley 值”或“留一法边际贡献”，而不能表述为精确 Shapley 计算。

```powershell
python APDP-RTFL/main.py --experiment-suite contribution --contribution-methods dp_fl,dp_rtfl,apdp_rtfl --contribution-quality-weight 0.25 --contribution-shapley-weight 0.35 --contribution-risk-weight 0.30 --contribution-fairness-weight 0.10 --contribution-utility-metric balanced_accuracy --enable-regulatory-intervention --run-name contribution_emnist_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --backend sklearn --seed 42
```

可使用 `contribution_penalty_summary.csv`、`approx_shapley_by_client.png`、`penalty_components.png` 和 `contribution_weight_alignment.png` 讨论数据质量、监管风险惩罚、公平性惩罚与聚合权重之间的平衡关系。

## 6. 审计溯源实验

该套件会自动记录按客户端、按轮次生成的 SHA-256 哈希链，并在正常 APDP-RTFL 训练过程中同步记录监管、公平性和贡献评估信息。

```powershell
python APDP-RTFL/main.py --experiment-suite audit_trace --audit-methods apdp_rtfl --audit-digest-algorithm sha256 --run-name audit_emnist_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --backend sklearn --seed 42
```

请报告 `audit_trace_summary.csv`、`audit_chain_verification.csv`、`audit_trace_log.csv` 和 `audit_trace_timeline.png`。正常完成的实验中，验证表应显示无效链路数为 0。

## 7. 模块消融实验

所有消融情景必须使用同一组随机种子，并报告相对于 `full` 的绝对变化和相对变化。

```powershell
python APDP-RTFL/main.py --experiment-suite ablation --ablation-method apdp_rtfl --ablation-scenarios full,no_adaptive_privacy,no_compute_adapter,no_zkip,no_ebcd,no_tcm,no_regulatory,no_contribution,no_fairness --run-name ablation_emnist_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --backend sklearn --seed 42
```

主要输出包括 `ablation_final_metrics.csv`、`ablation_summary.csv`、`ablation_accuracy.png`、`ablation_macro_f1.png`、`ablation_balanced_accuracy.png` 和 `ablation_accuracy_delta.png`。

## 论文结果矩阵建议

| 论文问题 | 主要实验套件 | 最低报告指标 |
| --- | --- | --- |
| DP 效用与隐私权衡 | `baselines` | Accuracy、Macro-F1、Balanced Accuracy、适用时的 AUC、噪声尺度、运行时间 |
| 及时监管干预 | 启用干预的 `baselines` | 预警数、降权数、隔离数、效用变化 |
| 数据污染识别 | `pollution` | 识别率、假阳性、假阴性、攻击下的模型效用 |
| 合成群体公平性 | `synthetic_fairness` | 最差群体准确率、Accuracy/F1 差距、epsilon 差距、参与差距 |
| 数据质量与贡献治理 | `contribution` | 近似 Shapley 值、风险/公平性惩罚、贡献与权重一致性 |
| 审计可追溯性 | `audit_trace` | 审计事件数、已验证链路、无效链路、按轮次审计字段 |
| 机制必要性 | `ablation` | 完整模型与移除模块后的性能差异 |

汇总多个随机种子前，应保留每一个原始运行目录。只在单独的分析目录中汇总最终指标 CSV，绝不能覆盖原始审计、监管或污染日志。

## 8. 多随机种子汇总与论文主表

每个正式命令应针对每一个 seed 单独运行，并保持稳定的命名模式。例如，主基线实验依次使用 `--run-name baseline_emnist_seed42`、`baseline_emnist_seed43` 和 `baseline_emnist_seed44`。

在不修改原始基线目录的前提下进行汇总：

```powershell
python APDP-RTFL/aggregate_results.py --input-root results --run-pattern baseline_emnist_seed* --input-file baseline_final_metrics.csv --output-dir results/baseline_emnist_aggregate --title-prefix "EMNIST DP Baselines"
```

汇总器会生成三个 CSV 文件：

| 输出文件 | 用途 |
| --- | --- |
| `experiment_seed_metrics.csv` | 每个原始运行目录中的每种方法一行，并从目录名称推断 seed。用于可追溯性和统计检查。 |
| `experiment_metric_summary.csv` | 长表形式的 `method-metric-mean-std-n` 汇总，可用于作图或补充材料。 |
| `experiment_paper_main_table.csv` | 宽表形式，包含 `<metric>_mean`、`<metric>_std` 和 `<metric>_n` 列，可作为论文主结果表的数据源。 |

该脚本还会生成带均值和样本标准差误差线的 `aggregate_<metric>.png` 图。其他实验套件只需将 `--input-file` 改为对应的最终指标文件，例如 `pollution_final_metrics.csv`、`ablation_final_metrics.csv` 或 `privacy_sensitivity_final_metrics.csv`，并指定独立的汇总输出目录。

默认汇总指标为最终 Accuracy、Macro-F1、Balanced Accuracy、AUC、平均每轮时间和平均 DP 噪声尺度。当表格目的更聚焦时，可指定所需指标：

```powershell
python APDP-RTFL/aggregate_results.py --input-root results --run-pattern pollution_label_flip_seed* --input-file pollution_final_metrics.csv --metrics final_accuracy,final_f1_score,detection_rate,false_positive_rate,false_negative_rate --output-dir results/pollution_label_flip_aggregate --title-prefix "Label-Flipping Detection"
```

# GRAIL-FL 实验命令手册（中文版）

本手册用于组织面向受监管行业的 GRAIL-FL 可复现实验。请在项目根目录执行全部命令：

```powershell
python APDP-RTFL/main.py <arguments>
```

结果路径为 `results/<run-name>_YYYYmmdd_HHMMSS/`。每次正式实验均应显式指定 `--run-name` 作为逻辑实验名前缀；即使已指定前缀，程序也会自动追加启动时间戳。同一张比较表或同一幅图中的所有方法，必须使用相同的数据集划分、客户端数量、训练轮数、数据划分方式、隐私预算参数、随机种子集合和后端。

每个带时间戳的运行目录均采用一次写入规则：若最终目录已存在，程序会拒绝运行，以防止不同实验产物混入同一目录。每次新运行都会生成 `run_config.json`、`run_command.txt`、`environment.json`、`data_artifacts/` 和 `artifact_manifest.csv`。其中 `data_artifacts/` 保存数据集指纹、客户端划分摘要和已生成的掉线计划；启用 TCM 的方法目录还会保存 `tcm_manifest.csv` 与可恢复的 `checkpoints/*.npz` 文件。

## 通用正式实验设置

以下参数是 EMNIST 正式实验的起始设置。该配置采用 non-IID 划分和固定随机种子。每一条正式命令至少应使用三个随机种子重复运行，例如 `42`、`43`、`44`，并报告均值和标准差。

```powershell
--dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --min-epsilon 0.1 --max-epsilon 2 --dp-epsilon 1 --dp-delta 1e-5 --dp-l2-norm-clip 1 --backend sklearn --seed 42
```

仅用于功能验证的小样本测试可附加 `--max-samples 500 --num-rounds 2 --num-clients 5`。小样本结果不得写入论文。

## GRAIL-FL：受监管资源—隐私联合编排

将 `--heterogeneity-profile regulated_generic` 用于启用 GRAIL-FL。该模式以可复现的受限、标准、高性能三档资源画像模拟计算吞吐、上行带宽、RTT 与在线波动；在不改变每客户端总 RDP 账本的前提下，联合选择本地 epoch、参数块上传比例和 DP-SGD 噪声。不要把此模拟表述为真实行业设备实测。

当前 GRAIL-FL 的资源—隐私编排包含三项核心机制：

1. **机会感知隐私支出调度**：不再使用简单的 `remaining_epsilon / remaining_rounds` 均摊策略，而是根据客户端未来有效参与机会、数据质量、历史贡献、监管风险和全局预算利用率决定本轮 DP-SGD 噪声乘子。低算力客户端不会被直接映射为更低隐私支出；若其未来可参与窗口较少且处于合规状态，系统会在其成功参与时给予更有效的预算支出。
2. **deadline slack 感知的部分参数上传**：full upload 若具有足够 deadline 裕量则保持完整上传；若 full upload 虽可行但过于贴近 deadline，则自动选择 `0.5` 或 `0.25` 参数块比例，以恢复安全时延裕量。该机制用于通信压力下的适配，而不是为了制造压缩效果而无条件降低上传量。
3. **残差感知误差反馈**：未上传参数块的残差会保留到客户端本地。若残差压力超过阈值且 full upload 仍满足 deadline，系统会触发 `residual_feedback_full_upload`，优先完整上传一次以释放误差反馈，降低长期部分更新对模型精度的损伤。

```powershell
python APDP-RTFL/main.py --experiment-suite baselines --methods grail_fl --run-name grail_emnist_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --backend sklearn --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

关键参数说明：

| 参数 | 含义 | 建议默认值 |
| --- | --- | --- |
| `--round-deadline-seconds` | 每轮同步训练的模拟 deadline | `5` |
| `--reference-batch-seconds` | 单位算力处理一个 mini-batch 的参考时长 | `0.01` |
| `--parameter-blocks` | 线性模型参数块数量 | `8` |
| `--upload-ratios` | 候选参数块上传比例 | `1.0,0.5,0.25` |
| `--arpa-privacy-boost-gain` | 预算利用率落后时的隐私支出 boost 强度 | `0.8` |
| `--arpa-max-privacy-boost` | 预算利用率 boost 上限 | `1.8` |
| `--arpa-opportunity-compensation-weight` | 低未来参与机会的隐私支出补偿权重 | `0.65` |
| `--arpa-compression-slack-target` | full upload 超过 deadline 该比例时尝试部分上传 | `0.85` |
| `--arpa-residual-full-upload-threshold` | 残差压力超过该值时优先完整上传 | `0.25` |

每个 GRAIL-FL 运行额外输出：

| 输出文件 | 用途 |
| --- | --- |
| `resource_profiles.csv` | 每个客户端的资源层级、算力、带宽、RTT 和在线概率。 |
| `resource_trace.csv` | 每轮每客户端的资源状态、deadline/privacy 可行性和选择状态。 |
| `orchestration_decisions.csv` | 被选客户端的 epoch、上传比例、噪声乘子、预算目标、机会补偿、deadline slack 和上传选择原因。 |
| `partial_update_metrics.csv` | 部分参数上传比例、真实参数覆盖率、残差压力、残差前后 L2 范数和是否触发误差反馈 full upload。 |
| `resource_privacy_diagnostics.csv` | 客户端级资源—隐私诊断，包括 epsilon 利用率、有效参与率、deadline 可行率、平均噪声和残差压力。 |
| `tier_privacy_summary.csv` | 按 constrained、standard、high 三档资源层汇总 epsilon 利用率、有效参与率、上传比例和残差反馈次数。 |
| `tier_epsilon_utilization.png` | 各资源层平均 epsilon 利用率图。 |
| `tier_effective_participation.png` | 各资源层有效参与率图。 |
| `tier_upload_ratio.png` | 各资源层平均上传比例图。 |

报告 GRAIL-FL 结果时，至少同时给出 balanced accuracy、macro-F1、平均 epsilon 利用率、低资源层有效参与率、deadline 达成率、平均上传比例和残差反馈次数。若 GRAIL-FL 未优于 DP 基线，应按预设协议报告，不得通过提高 GRAIL-FL 的总隐私预算或放宽其资源条件获取优势。

## 1. DP 基线对照

正式 DP 基线实验在准备论文最终主表时必须显式列出方法。本文更新后的方法名称为 GRAIL-FL，并且必须同时包含 `--heterogeneity-profile regulated_generic`。

主 DP 对照组为：

`DP-FedAvg`、`DP-FedProx`、`DP-FedSGD`、`DP-FedNova` 和本文方法 `GRAIL-FL`。`dp_fl`、`dp_flprox`、`ldp_fl`、`global_dp`、`dp_rtfl`、`apdp_rtfl` 和 `dp_fedadam` 等旧方法键仍可用于复核历史实验，但不再纳入论文最终主表。

## 客户端 DP-SGD 与隐私会计

客户端 DP 方法采用样本级本地 DP-SGD：每个客户端在本地按 Poisson mini-batch 进行逐样本梯度裁剪和高斯加噪，服务器不额外注入聚合端 DP 噪声。`--epsilon-per-client-total` 表示**每个客户端整个训练过程**的目标预算，默认值为 5，`--dp-delta` 默认 `1e-5`，并由 RDP 会计器按实际 DP-SGD 步数累计。旧参数 `--total-privacy-budget` 仅作为该参数的兼容别名，不再表示可在客户端之间分割并在每轮重置的共享预算。

默认 DP-SGD 的期望本地 batch size 为 256，可通过 `--dp-batch-size` 修改。每次客户端 DP 运行都会输出 `privacy_accounting.csv` 与 `privacy_accounting_summary.csv`；报告时应给出每个客户端的最终 epsilon、目标 epsilon 和预算耗尽事件数。

## PyTorch/GPU 后端

使用 `--backend torch --device cuda` 时，客户端模型、本地训练张量、逐样本梯度裁剪和高斯 DP-SGD 加噪都会在 GPU 上执行。默认 `--torch-model linear` 保留历史线性 softmax/logistic 模型；若要诊断模型容量瓶颈，可使用 `--torch-model mlp --torch-mlp-hidden 256,128` 启用小型全连接网络。当 torch 后端与 `--heterogeneity-profile regulated_generic` 结合使用时，会进入与 sklearn 相同的完整 GRAIL-FL runner：资源编排、客户端选择、隐私支出、部分上传、RDP 会计、ZKIP/EBCD/TCM、监管干预、贡献评分、审计溯源和机制诊断产物都保持同一语义。正式 CUDA 论文实验应同时使用 `--dp-batch-size 256 --torch-batch-size 256` 与受监管资源 profile。

在实验服务器上使用该后端前，应先安装支持 CUDA 的 PyTorch，并验证设备可用：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

该命令应输出 `True`。

### S2：模型容量与隐私强度诊断

在继续正式主表实验前，应先用同一数据划分、同一客户端数和同一模型骨干做三组诊断。该阶段的目的不是生成论文主表，而是判断当前 `0.5-0.6` 精度主要来自模型容量不足、DP 噪声过强，还是 GRAIL-FL 调度策略。为节省计算资源，S2 诊断阶段只使用一组随机种子，默认 `seed=42`；只有当某组配置表现合理并准备进入正式主表时，再扩展到 `42/43/44` 多随机种子。建议优先在 EMNIST balanced 上执行，若 FEMNIST 仍异常，再单独降低 FEMNIST 难度做补充诊断。

诊断命名规则：

| 诊断组 | 目的 | 方法 | 隐私预算 | 推荐前缀 |
| --- | --- | --- | --- | --- |
| no-DP 上限 | 判断 MLP 在当前 FL 划分下的可达上限 | `fedavg,fedprox` | 不使用客户端 DP | `s2_upper_${dataset}_seed42` |
| 弱 DP | 判断较宽松 DP 下是否接近 no-DP 上限 | `dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl` | `epsilon_per_client_total=20` | `s2_weakdp_${dataset}_seed42` |
| 强 DP | 判断正式预算下的可发表性能 | `dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl` | `epsilon_per_client_total=5` | `s2_strongdp_${dataset}_seed42` |

no-DP 上限诊断：

```bash
python APDP-RTFL/main.py \
  --experiment-suite baselines \
  --methods fedavg,fedprox \
  --run-name s2_upper_emnist_seed42 \
  --dataset emnist \
  --emnist-split balanced \
  --num-clients 20 \
  --num-rounds 200 \
  --client-epochs 3 \
  --partition dirichlet \
  --dirichlet-alpha 0.5 \
  --epsilon-per-client-total 5 \
  --dp-batch-size 256 \
  --torch-batch-size 256 \
  --backend torch \
  --device cuda \
  --torch-model mlp \
  --torch-mlp-hidden 256,128 \
  --heterogeneity-profile regulated_generic \
  --failure-prob 0 \
  --seed 42
```

弱 DP 诊断：

```bash
python APDP-RTFL/main.py \
  --experiment-suite baselines \
  --methods dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl \
  --run-name s2_weakdp_emnist_seed42 \
  --dataset emnist \
  --emnist-split balanced \
  --num-clients 20 \
  --num-rounds 200 \
  --client-epochs 3 \
  --partition dirichlet \
  --dirichlet-alpha 0.5 \
  --epsilon-per-client-total 20 \
  --min-epsilon 0.1 \
  --max-epsilon 4 \
  --dp-epsilon 1 \
  --dp-delta 1e-5 \
  --dp-l2-norm-clip 1 \
  --dp-batch-size 256 \
  --torch-batch-size 256 \
  --backend torch \
  --device cuda \
  --torch-model mlp \
  --torch-mlp-hidden 256,128 \
  --heterogeneity-profile regulated_generic \
  --round-deadline-seconds 5 \
  --reference-batch-seconds 0.01 \
  --parameter-blocks 8 \
  --upload-ratios 1.0,0.5,0.25 \
  --arpa-privacy-boost-gain 0.5 \
  --arpa-max-privacy-boost 1.5 \
  --arpa-opportunity-compensation-weight 0.65 \
  --arpa-compression-slack-target 0.85 \
  --arpa-residual-full-upload-threshold 0.25 \
  --failure-prob 0 \
  --seed 42
```

强 DP 诊断：

```bash
python APDP-RTFL/main.py \
  --experiment-suite baselines \
  --methods dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl \
  --run-name s2_strongdp_emnist_seed42 \
  --dataset emnist \
  --emnist-split balanced \
  --num-clients 20 \
  --num-rounds 200 \
  --client-epochs 3 \
  --partition dirichlet \
  --dirichlet-alpha 0.5 \
  --epsilon-per-client-total 5 \
  --min-epsilon 0.1 \
  --max-epsilon 2 \
  --dp-epsilon 1 \
  --dp-delta 1e-5 \
  --dp-l2-norm-clip 1 \
  --dp-batch-size 256 \
  --torch-batch-size 256 \
  --backend torch \
  --device cuda \
  --torch-model mlp \
  --torch-mlp-hidden 256,128 \
  --heterogeneity-profile regulated_generic \
  --round-deadline-seconds 5 \
  --reference-batch-seconds 0.01 \
  --parameter-blocks 8 \
  --upload-ratios 1.0,0.5,0.25 \
  --arpa-privacy-boost-gain 0.5 \
  --arpa-max-privacy-boost 1.5 \
  --arpa-opportunity-compensation-weight 0.65 \
  --arpa-compression-slack-target 0.85 \
  --arpa-residual-full-upload-threshold 0.25 \
  --failure-prob 0 \
  --seed 42
```

解释规则：

- 若 no-DP 上限仍低于 0.70，说明主要瓶颈不是 DP，而是模型容量、数据划分或训练强度；应先调整模型或任务设置。
- 若 no-DP 明显高于 0.70，而弱 DP 接近 no-DP，说明模型容量已经足够，正式预算下的精度损失主要来自 DP 噪声。
- 若弱 DP 明显高于强 DP，优先调 `dp-l2-norm-clip`、`dp-batch-size`、本地 epoch 和 GRAIL-FL 隐私调度，而不是放宽正式主表预算。
- 若 GRAIL-FL 在弱 DP 和强 DP 中都低于固定 DP 基线，应先降低 `arpa-privacy-boost-gain` 和 `arpa-max-privacy-boost`，避免预算利用率提升被更大噪声抵消。
- S2 诊断结果只能用于选择正式实验配置；正式主表仍应使用同一模型骨干、同一隐私预算、同一数据划分和同一 seed 集。

torch/GPU-GRAIL 四组论文数据集基线命令如下：

```bash
for dataset in emnist femnist cifar10 medmnist; do
  for seed in 42 43 44; do
    python APDP-RTFL/main.py \
      --experiment-suite baselines \
      --methods dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl \
      --run-name grail_main_${dataset}_seed${seed} \
      --dataset ${dataset} \
      --emnist-split balanced \
      --num-clients 20 \
      --num-rounds 200 \
      --client-epochs 3 \
      --partition dirichlet \
      --dirichlet-alpha 0.5 \
      --epsilon-per-client-total 5 \
      --min-epsilon 0.1 \
      --max-epsilon 2 \
      --dp-epsilon 1 \
      --dp-delta 1e-5 \
      --dp-l2-norm-clip 1 \
      --dp-batch-size 256 \
      --torch-batch-size 256 \
      --backend torch \
      --device cuda \
      --heterogeneity-profile regulated_generic \
      --round-deadline-seconds 5 \
      --reference-batch-seconds 0.01 \
      --parameter-blocks 8 \
      --upload-ratios 1.0,0.5,0.25 \
      --arpa-privacy-boost-gain 0.8 \
      --arpa-max-privacy-boost 1.8 \
      --arpa-opportunity-compensation-weight 0.65 \
      --arpa-compression-slack-target 0.85 \
      --arpa-residual-full-upload-threshold 0.25 \
      --seed ${seed}
  done
done
```

CUDA 是实验加速后端，不表示真实边缘客户端都具备 GPU。资源异构仍由 regulated resource profile 与 deadline simulation 控制。下方 sklearn/GRAIL 命令继续作为 CPU 兼容的参考路径，可用于后端一致性检查。

```bash
for dataset in emnist femnist cifar10 medmnist; do
  for seed in 42 43 44; do
    python APDP-RTFL/main.py \
      --experiment-suite baselines \
      --methods dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl \
      --run-name grail_main_${dataset}_seed${seed} \
      --dataset ${dataset} \
      --emnist-split balanced \
      --num-clients 20 \
      --num-rounds 200 \
      --client-epochs 3 \
      --partition dirichlet \
      --dirichlet-alpha 0.5 \
      --epsilon-per-client-total 5 \
      --min-epsilon 0.1 \
      --max-epsilon 2 \
      --dp-epsilon 1 \
      --dp-delta 1e-5 \
      --dp-l2-norm-clip 1 \
      --backend sklearn \
      --heterogeneity-profile regulated_generic \
      --round-deadline-seconds 5 \
      --reference-batch-seconds 0.01 \
      --parameter-blocks 8 \
      --upload-ratios 1.0,0.5,0.25 \
      --arpa-privacy-boost-gain 0.8 \
      --arpa-max-privacy-boost 1.8 \
      --arpa-opportunity-compensation-weight 0.65 \
      --arpa-compression-slack-target 0.85 \
      --arpa-residual-full-upload-threshold 0.25 \
      --seed ${seed}
  done
done
```

主要输出为 `baseline_final_metrics.csv`、`baseline_summary.csv`、`baseline_comparison.png` 和 `baseline_method_metadata.csv`。其中元数据文件保存了论文比较表对应的代码侧配置记录：

| 方法 | 项目内配置 | 参考文献 |
| --- | --- | --- |
| DP-FedAvg (`dp_fedavg`) | 客户端侧 DP FedAvg | McMahan et al., AISTATS 2017；McMahan et al., ICLR 2018。 |
| DP-FedProx (`dp_fedprox`) | 客户端侧 DP 更新加 FedProx 近端项 | Li et al., *Federated optimization in heterogeneous networks*, MLSys, 2020。 |
| DP-FedSGD (`dp_fedsgd`) | 客户端侧 DP 加强制单轮本地更新的 FedSGD 风格对照 | McMahan et al., ICLR, 2018；作为 FedSGD 风格单步更新基线。 |
| DP-FedNova (`dp_fednova`) | 面向异构本地步数的客户端侧 DP 归一化聚合 | Wang et al., *Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization*, NeurIPS, 2020。 |
| GRAIL-FL (`grail_fl`) | 客户端侧 DP 加受监管资源—隐私—治理联合编排 | 本文所提方法；必须启用 `regulated_generic`。 |

旧基线仍可用于复现实验和历史结果复核，但论文最终主表应只使用上方五种方法。

### 其他 CUDA 实验套件

参与策略对比：

```powershell
python APDP-RTFL/main.py --experiment-suite participation --participation-policies all,random,apdp_score --participation-rate 0.6 --run-name participation_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

隐私预算敏感性实验：

```powershell
python APDP-RTFL/main.py --experiment-suite privacy_sensitivity --privacy-budgets 20,50,80,100 --privacy-sensitivity-methods dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl --run-name privacy_sensitivity_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

客户端公平性实验：

```powershell
python APDP-RTFL/main.py --experiment-suite fairness --fairness-methods dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl --run-name fairness_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

## 2. 监管干预实验

该实验用于量化预警、降权、隔离及其对模型效用的影响。除监管开关外，其他参数应与 DP 基线命令完全一致。

```powershell
python APDP-RTFL/main.py --experiment-suite baselines --methods grail_fl --enable-regulatory-intervention --reg-warning-threshold 1.5 --reg-quarantine-threshold 2.5 --reg-penalty-weight 0.5 --run-name regulatory_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

应将该结果与去掉 `--enable-regulatory-intervention` 的同一命令进行对照。主要输出包括 `regulatory_intervention_summary.csv`、`regulatory_actions.png` 和 `regulatory_risk_by_client.png`。

## 3. 数据污染识别与监管干预

普通训练不会启用污染。只有同时使用 `--experiment-suite pollution` 与 `--enable-pollution-injection` 时，污染注入才会生效。

标签翻转场景：

```powershell
python APDP-RTFL/main.py --experiment-suite pollution --methods grail_fl --enable-pollution-injection --pollution-type label_flip --polluted-clients 1,3 --pollution-start-round 10 --enable-regulatory-intervention --run-name pollution_label_flip_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

特征噪声场景：

```powershell
python APDP-RTFL/main.py --experiment-suite pollution --methods grail_fl --enable-pollution-injection --pollution-type feature_noise --polluted-clients 1,3 --pollution-start-round 10 --enable-regulatory-intervention --run-name pollution_feature_noise_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

请从 `pollution_final_metrics.csv`、`pollution_summary.csv`、`pollution_injection_summary.csv` 和 `pollution_detection_rate.png` 中报告效用变化、识别率、假阳性率和假阴性率。

## 4. 合成敏感属性公平性压力测试

该实验属于客户端级的合成压力测试，而非对真实人口统计公平性的验证。它构造样本覆盖、标签分布、特征质量、参与稳定性和计算能力之间存在关联的群体差异。

```powershell
python APDP-RTFL/main.py --experiment-suite synthetic_fairness --fairness-datasets emnist --fairness-methods dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl --synthetic-sensitive-attrs gender,age,region --fairness-pressure-profile regulated --run-name synthetic_fairness_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

对于 FEMNIST、CIFAR10 和 CIFAR100，应先生成 `data/<dataset>/all_data`，再将 `--fairness-datasets emnist` 替换为所需数据集的逗号分隔列表。主要输出包括 `synthetic_sensitive_clients.csv`、`synthetic_group_fairness_summary.csv`、`federated_group_fairness_summary.csv` 及群体公平性图表。

论文建议表述：本文使用客户端级合成敏感属性模拟机构异质性与资源不均衡，该设置不代表真实的性别、年龄或地区人口统计属性。

## 5. 惩罚机制与近似 Shapley 贡献评估

当前贡献评估器以留一法边际效用计算近似 Shapley 值。论文中应明确称其为“近似 Shapley 值”或“留一法边际贡献”，而不能表述为精确 Shapley 计算。

```powershell
python APDP-RTFL/main.py --experiment-suite contribution --contribution-methods dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl --contribution-quality-weight 0.25 --contribution-shapley-weight 0.35 --contribution-risk-weight 0.30 --contribution-fairness-weight 0.10 --contribution-utility-metric balanced_accuracy --enable-regulatory-intervention --run-name contribution_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

可使用 `contribution_penalty_summary.csv`、`approx_shapley_by_client.png`、`penalty_components.png` 和 `contribution_weight_alignment.png` 讨论数据质量、监管风险惩罚、公平性惩罚与聚合权重之间的平衡关系。

## 6. 审计溯源实验

该套件会自动记录按客户端、按轮次生成的 SHA-256 哈希链，并在正常 GRAIL-FL 训练过程中同步记录监管、公平性和贡献评估信息。

```powershell
python APDP-RTFL/main.py --experiment-suite audit_trace --audit-methods dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl --audit-digest-algorithm sha256 --run-name audit_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

请报告 `audit_trace_summary.csv`、`audit_chain_verification.csv`、`audit_trace_log.csv` 和 `audit_trace_timeline.png`。正常完成的实验中，验证表应显示无效链路数为 0。

## 7. 模块消融实验

所有消融情景必须使用同一组随机种子，并报告相对于 `full` 的绝对变化和相对变化。

```powershell
python APDP-RTFL/main.py --experiment-suite ablation --ablation-method grail_fl --ablation-scenarios full,no_adaptive_privacy,no_compute_adapter,no_resource_orchestration,no_partial_updates,no_resource_fairness,no_opportunity_privacy,no_budget_utilization_boost,no_low_resource_compensation,no_zkip,no_ebcd,no_tcm,no_regulatory,no_contribution,no_fairness --run-name ablation_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

主要输出包括 `ablation_final_metrics.csv`、`ablation_summary.csv`、`ablation_accuracy.png`、`ablation_macro_f1.png`、`ablation_balanced_accuracy.png` 和 `ablation_accuracy_delta.png`。

新增 GRAIL-FL 相关消融解释如下：

| 消融场景 | 移除内容 | 主要观察指标 |
| --- | --- | --- |
| `no_resource_orchestration` | 移除资源—隐私联合编排路径 | balanced accuracy、deadline 达成率、慢节点参与率 |
| `no_partial_updates` | 强制 full upload，移除参数块部分更新 | 上传字节数、deadline 达成率、精度 |
| `no_resource_fairness` | 移除资源层轮换覆盖约束 | constrained/standard/high 参与差距 |
| `no_opportunity_privacy` | 移除未来有效参与机会感知预算调度 | epsilon 利用率、低资源层噪声乘子、精度 |
| `no_budget_utilization_boost` | 移除预算利用率闭环 boost | 最终 epsilon 利用率、收敛速度、精度 |
| `no_low_resource_compensation` | 移除低未来参与机会客户端的隐私支出补偿 | constrained 层 epsilon 利用率与有效参与率 |

消融报告不应只比较最终 accuracy。应同时报告 `tier_privacy_summary.csv` 中的资源层 epsilon 利用率、有效参与率、上传比例和残差反馈次数，以说明性能变化是否来自低资源客户端被重新纳入有效训练。

## 论文结果矩阵建议

| 论文问题 | 主要实验套件 | 最低报告指标 |
| --- | --- | --- |
| DP 效用与隐私权衡 | `baselines` | Accuracy、Macro-F1、Balanced Accuracy、适用时的 AUC、噪声尺度、运行时间 |
| 资源—隐私联合编排 | `baselines` + `heterogeneity_profile=regulated_generic` | 资源层 epsilon 利用率、deadline 达成率、上传比例、残差反馈次数、慢节点有效参与率 |
| 及时监管干预 | 启用干预的 `baselines` | 预警数、降权数、隔离数、效用变化 |
| 数据污染识别 | `pollution` | 识别率、假阳性、假阴性、攻击下的模型效用 |
| 合成群体公平性 | `synthetic_fairness` | 最差群体准确率、Accuracy/F1 差距、epsilon 差距、参与差距 |
| 数据质量与贡献治理 | `contribution` | 近似 Shapley 值、风险/公平性惩罚、贡献与权重一致性 |
| 审计可追溯性 | `audit_trace` | 审计事件数、已验证链路、无效链路、按轮次审计字段 |
| 机制必要性 | `ablation` | 完整模型与移除模块后的性能差异 |

汇总多个随机种子前，应保留每一个原始运行目录。只在单独的分析目录中汇总最终指标 CSV，绝不能覆盖原始审计、监管或污染日志。

## 8. 多随机种子汇总与论文主表

每个正式命令应针对每一个 dataset 和 seed 单独运行，并保持稳定的命名前缀。新版 GRAIL-FL 主基线实验建议使用 `grail_main_emnist_seed42`、`grail_main_femnist_seed42`、`grail_main_cifar10_seed42`、`grail_main_medmnist_seed42` 等名称；程序会自动追加时间戳。准备论文最终主表时不要再使用旧 APDP 结果前缀。

在不修改原始基线目录的前提下进行汇总：

```bash
for dataset in emnist femnist cifar10 medmnist; do
  python APDP-RTFL/aggregate_results.py \
    --input-root results \
    --run-pattern grail_main_${dataset}_seed* \
    --input-file baseline_final_metrics.csv \
    --output-dir results/grail_main_${dataset}_aggregate \
    --title-prefix "${dataset} DP Baselines with GRAIL-FL"
done
```

汇总器会生成三个 CSV 文件：

| 输出文件 | 用途 |
| --- | --- |
| `experiment_seed_metrics.csv` | 每个原始运行目录中的每种方法一行，并从目录名称推断 seed。用于可追溯性和统计检查。 |
| `experiment_metric_summary.csv` | 长表形式的 `method-metric-mean-std-n` 汇总，可用于作图或补充材料。 |
| `experiment_paper_main_table.csv` | 宽表形式，包含 `<metric>_mean`、`<metric>_std` 和 `<metric>_n` 列，可作为论文主结果表的数据源。 |

该脚本还会生成带均值和样本标准差误差线的 `aggregate_<metric>.png` 图。其他实验套件只需将 `--input-file` 改为对应的最终指标文件，例如 `pollution_final_metrics.csv`、`ablation_final_metrics.csv` 或 `privacy_sensitivity_final_metrics.csv`，并指定独立的汇总输出目录。

GRAIL-FL 的资源层诊断需要单独汇总。该脚本读取每个运行目录下的 `grail_fl/tier_privacy_summary.csv` 和 `grail_fl/resource_privacy_diagnostics.csv`；若输入是消融实验目录，也会递归读取 `scenario/grail_fl/` 下的诊断文件：

```bash
for dataset in emnist femnist cifar10 medmnist; do
  python APDP-RTFL/aggregate_arpa_diagnostics.py \
    --input-root results \
    --run-pattern grail_main_${dataset}_seed* \
    --output-dir results/grail_${dataset}_diagnostics_aggregate \
    --title-prefix "${dataset} GRAIL-FL Diagnostics"
done
```

正式验收时建议加入 `--require-complete`，使任何缺少 `tier_privacy_summary.csv` 的运行目录都会导致汇总失败，避免旧格式或失败运行被静默跳过：

```bash
for dataset in emnist femnist cifar10 medmnist; do
  python APDP-RTFL/aggregate_arpa_diagnostics.py \
    --input-root results \
    --run-pattern grail_main_${dataset}_seed* \
    --require-complete \
    --output-dir results/grail_${dataset}_diagnostics_aggregate_strict \
    --title-prefix "${dataset} GRAIL-FL Diagnostics"
done
```

主要输出包括：

| 输出文件 | 用途 |
| --- | --- |
| `arpa_tier_seed_metrics.csv` | 每个原始运行目录、每个资源层一行，保留 seed、scenario、method 和 tier。 |
| `arpa_tier_metric_summary.csv` | 长表形式的资源层指标均值、标准差和样本数。 |
| `arpa_tier_paper_table.csv` | 宽表形式，可直接整理为论文机制诊断表。 |
| `arpa_client_seed_diagnostics.csv` | 合并后的客户端级资源—隐私诊断明细。 |
| `aggregate_tier_<metric>_<scenario>.png` | 各资源层指标的均值和标准差图。 |

默认汇总指标包括 `avg_epsilon_utilization`、`avg_historical_success_rate`、`avg_deadline_feasible_rate`、`avg_noise_multiplier`、`avg_upload_ratio`、`avg_residual_pressure`、`compressed_selection_count` 和 `residual_feedback_full_upload_count`。若只关注论文主文中的核心治理指标，可显式指定：

```bash
for dataset in emnist femnist cifar10 medmnist; do
  python APDP-RTFL/aggregate_arpa_diagnostics.py \
    --input-root results \
    --run-pattern grail_main_${dataset}_seed* \
    --metrics avg_epsilon_utilization,avg_historical_success_rate,avg_upload_ratio,residual_feedback_full_upload_count \
    --output-dir results/grail_${dataset}_diagnostics_core \
    --title-prefix "${dataset} GRAIL-FL Core Diagnostics"
done
```

如果只关注低资源层，可使用 `--tiers constrained`；如果输入目录是消融实验，可用 `--scenario-filter full,no_partial_updates` 只汇总指定场景：

```powershell
python APDP-RTFL/aggregate_arpa_diagnostics.py --input-root results --run-pattern ablation_emnist_seed*_grail* --scenario-filter full,no_partial_updates,no_opportunity_privacy --tiers constrained --output-dir results/grail_ablation_constrained_diagnostics --title-prefix "Constrained-tier GRAIL-FL Ablation"
```

默认汇总指标为最终 Accuracy、Macro-F1、Balanced Accuracy、AUC、平均每轮时间和平均 DP 噪声尺度。当表格目的更聚焦时，可指定所需指标：

```powershell
python APDP-RTFL/aggregate_results.py --input-root results --run-pattern pollution_label_flip_seed*_grail* --input-file pollution_final_metrics.csv --metrics final_accuracy,final_f1_score,detection_rate,false_positive_rate,false_negative_rate --output-dir results/pollution_label_flip_grail_aggregate --title-prefix "Label-Flipping Detection with GRAIL-FL"
```

## 9. GRAIL-FL 单次运行验收检查

完成包含 GRAIL-FL 与 DP 基线的方法对照后，可使用验收脚本检查该运行是否满足预设条件。该脚本不会改变原始训练产物，只在独立输出目录中写入检查表。

```powershell
python APDP-RTFL/validate_arpa_acceptance.py --run-dir results/grail_main_emnist_seed42_YYYYmmdd_HHMMSS --metric final_balanced_accuracy --baselines dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova --require-all-baselines --min-win-margin 0.0 --min-epsilon-utilization 0.70 --min-constrained-success-rate 0.50 --max-deadline-failure-rate 0.20 --output-dir results/grail_main_emnist_seed42_acceptance
```

验收项包括：

| 验收项 | 含义 |
| --- | --- |
| `requested_baselines_present` | 指定基线是否全部存在；正式验收建议使用 `--require-all-baselines`。 |
| `beats_present_baselines` | GRAIL-FL 在指定主指标上是否超过当前运行中实际存在的 DP 基线。 |
| `avg_tier_epsilon_utilization` | 各资源层平均 epsilon 利用率是否达到阈值。 |
| `constrained_effective_participation` | constrained 层有效参与率是否达到阈值。 |
| `predicted_deadline_failure_rate` | 被选客户端的预测 deadline 失败率是否低于阈值。 |

输出文件包括 `arpa_acceptance_checks.csv`、`arpa_baseline_comparisons.csv` 和 `arpa_acceptance_summary.json`。比较表会记录 GRAIL-FL 与每个基线的指标差值 `margin`；若希望 GRAIL-FL 至少高出某个幅度，可设置 `--min-win-margin`。正式验收可加 `--strict`，要求 `grail_fl/tier_privacy_summary.csv` 与 `grail_fl/resource_trace.csv` 必须存在。

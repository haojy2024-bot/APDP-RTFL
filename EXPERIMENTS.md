# GRAIL-FL Experiment Command Guide

This guide defines reproducible commands for the regulated-industry GRAIL-FL study. Run every command from the repository root:

```powershell
python APDP-RTFL/main.py <arguments>
```

The result location is `results/<run-name>_YYYYmmdd_HHMMSS/`. Set `--run-name` explicitly as a logical experiment-name prefix for every formal run; the runner always appends its launch timestamp, including when a prefix is supplied. Use the same dataset split, client count, round count, partition, privacy-budget parameters, seed list, and backend within one comparison table or figure.

Each timestamped run directory is write-once: an existing final directory is rejected to prevent mixed artifacts. Every new run writes `run_config.json`, `run_command.txt`, `environment.json`, `data_artifacts/`, and `artifact_manifest.csv`. The data-artifact directory contains dataset fingerprints, client split summaries, and the generated failure plan; method directories with TCM enabled also contain `tcm_manifest.csv` and recoverable `checkpoints/*.npz` files.

## Common Formal Setting

The following is a starting protocol for a formal EMNIST experiment. It intentionally uses a non-IID partition and a fixed random seed. Repeat each formal command with at least three seeds, such as `42`, `43`, and `44`, and report mean plus standard deviation.

```powershell
--dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --min-epsilon 0.1 --max-epsilon 2 --dp-epsilon 1 --dp-delta 1e-5 --dp-l2-norm-clip 1 --backend sklearn --seed 42
```

For small smoke tests only, add `--max-samples 500 --num-rounds 2 --num-clients 5`. Do not use these small-sample results in the paper.

## GRAIL-FL: Regulated Resource-Privacy Orchestration

Use `--heterogeneity-profile regulated_generic` to enable GRAIL-FL. This mode uses reproducible constrained, standard, and high-resource client profiles to simulate compute throughput, uplink bandwidth, RTT, and online volatility. Without changing the per-client total RDP ledger, it jointly selects local epochs, parameter-block upload ratio, and DP-SGD noise. Do not describe this simulation as a real-world industry-device measurement.

The current GRAIL-FL resource-privacy orchestration contains three core mechanisms:

1. **Opportunity-aware privacy spending**: instead of simply using `remaining_epsilon / remaining_rounds`, the scheduler considers future effective participation opportunities, data quality, historical contribution, regulatory risk, and global budget utilization when choosing the current round DP-SGD noise multiplier. Low-resource clients are not directly mapped to lower privacy spending; when they have fewer future participation windows and remain compliant, the system can spend privacy budget more effectively when they successfully participate.
2. **Deadline-slack-aware partial parameter upload**: full upload is kept when it has enough deadline slack. If full upload is feasible but too close to the deadline, the scheduler selects a `0.5` or `0.25` parameter-block ratio to restore timing slack. This mechanism adapts to communication pressure; it is not an unconditional compression trick.
3. **Residual-aware error feedback**: parameter blocks that are not uploaded remain as local residuals. If residual pressure exceeds the threshold and full upload is still deadline-feasible, `residual_feedback_full_upload` is triggered to release accumulated error and reduce long-run accuracy loss from partial updates.

```powershell
python APDP-RTFL/main.py --experiment-suite baselines --methods grail_fl --run-name grail_emnist_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --backend sklearn --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

Key parameters:

| Parameter | Meaning | Suggested default |
| --- | --- | --- |
| `--round-deadline-seconds` | Simulated synchronous-round deadline | `5` |
| `--reference-batch-seconds` | Reference duration for processing one mini-batch at unit compute speed | `0.01` |
| `--parameter-blocks` | Number of linear-model parameter blocks | `8` |
| `--upload-ratios` | Candidate parameter-block upload ratios | `1.0,0.5,0.25` |
| `--arpa-privacy-boost-gain` | Privacy-spend boost gain when budget utilization lags | `0.8` |
| `--arpa-max-privacy-boost` | Upper bound for privacy-utilization boost | `1.8` |
| `--arpa-opportunity-compensation-weight` | Compensation weight for clients with scarce future participation opportunities | `0.65` |
| `--arpa-compression-slack-target` | Try partial upload when full upload exceeds this deadline fraction | `0.85` |
| `--arpa-residual-full-upload-threshold` | Prefer full upload when residual pressure exceeds this value | `0.25` |

Each GRAIL-FL run additionally outputs:

| Output file | Use |
| --- | --- |
| `resource_profiles.csv` | Resource tier, compute speed, bandwidth, RTT, and online probability for each client. |
| `resource_trace.csv` | Per-round per-client resource state, deadline/privacy feasibility, and selection status. |
| `orchestration_decisions.csv` | Selected clients' epochs, upload ratios, noise multipliers, budget targets, opportunity compensation, deadline slack, and upload-selection reasons. |
| `partial_update_metrics.csv` | Partial upload ratio, actual parameter coverage, residual pressure, residual L2 before/after, and whether error-feedback full upload was triggered. |
| `resource_privacy_diagnostics.csv` | Client-level resource-privacy diagnostics, including epsilon utilization, effective participation rate, deadline feasibility rate, average noise, and residual pressure. |
| `tier_privacy_summary.csv` | Resource-tier summary for constrained, standard, and high clients: epsilon utilization, effective participation rate, upload ratio, and residual-feedback count. |
| `tier_epsilon_utilization.png` | Mean epsilon utilization by resource tier. |
| `tier_effective_participation.png` | Effective participation rate by resource tier. |
| `tier_upload_ratio.png` | Mean upload ratio by resource tier. |

When reporting GRAIL-FL results, include at least balanced accuracy, macro-F1, average epsilon utilization, low-resource-tier effective participation rate, deadline success rate, average upload ratio, and residual-feedback count. If GRAIL-FL does not outperform a DP baseline, report it according to the pre-specified protocol; do not obtain an advantage by raising GRAIL-FL's total privacy budget or relaxing its resource constraints.

## 1. DP Baseline Comparison

Formal DP baseline experiments must list methods explicitly instead of using `--methods all` when preparing final paper tables. In this study, the updated method name is GRAIL-FL, and it must include `--heterogeneity-profile regulated_generic`.

The main DP comparison set is:

`DP-FedAvg`, `DP-FedProx`, `DP-FedNova`, `DP-FedAdam`, and the proposed `GRAIL-FL`. Legacy methods such as `dp_fl`, `dp_flprox`, `dp_fedsgd`, `ldp_fl`, `global_dp`, `dp_rtfl`, and `apdp_rtfl` remain explicitly runnable for older-result audits, but they should not be placed in the final main table.

## Client-Side DP-SGD And Privacy Accounting

Client-DP methods use sample-level local DP-SGD: each client runs Poisson mini-batches locally with per-sample gradient clipping and Gaussian noise, while the server does not add new aggregation-side DP noise. `--epsilon-per-client-total` is the target budget for each client's complete training trace, with default epsilon 5 and default `--dp-delta 1e-5`; the RDP accountant accumulates the actual DP-SGD steps. The old `--total-privacy-budget` option is only a compatibility alias for this parameter and no longer means a shared budget divided between clients or reset every round.

The expected local DP-SGD batch size is 256 and can be changed through `--dp-batch-size`. Every client-DP run writes `privacy_accounting.csv` and `privacy_accounting_summary.csv`; report each client's final epsilon, target epsilon, and number of budget-exhausted events.

## PyTorch/GPU Backend

Use `--backend torch --device cuda` to run the linear softmax/logistic client model, local client training tensors, per-sample gradient clipping, and Gaussian DP-SGD noise generation on the GPU. When combined with `--heterogeneity-profile regulated_generic`, torch uses the same full GRAIL-FL runner as sklearn across the paper experiment suites: resource orchestration, client selection, privacy spending, partial upload, RDP accounting, ZKIP/EBCD/TCM, regulatory intervention, contribution scoring, audit traceability, and mechanism diagnostics keep the same semantics. For CUDA paper runs, use `--dp-batch-size 256 --torch-batch-size 256` together with the regulated resource profile.

Install a CUDA-enabled PyTorch build on the experiment server before using this backend, then verify the device:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

The command should print `True`.

Torch/GPU-GRAIL baseline command:

```powershell
python APDP-RTFL/main.py --experiment-suite baselines --methods dp_fedavg,dp_fedprox,dp_fednova,dp_fedadam,grail_fl --run-name grail_main_emnist_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --min-epsilon 0.1 --max-epsilon 2 --dp-epsilon 1 --dp-delta 1e-5 --dp-l2-norm-clip 1 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

CUDA is an experimental execution backend, not a claim that all real edge clients have GPUs. Resource heterogeneity remains controlled by the regulated resource profile and deadline simulation. The sklearn/GRAIL command below remains available as a CPU-compatible reference path for backend consistency checks.

```powershell
python APDP-RTFL/main.py --experiment-suite baselines --methods dp_fedavg,dp_fedprox,dp_fednova,dp_fedadam,grail_fl --run-name grail_main_emnist_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --min-epsilon 0.1 --max-epsilon 2 --dp-epsilon 1 --dp-delta 1e-5 --dp-l2-norm-clip 1 --backend sklearn --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

Key outputs are `baseline_final_metrics.csv`, `baseline_summary.csv`, `baseline_comparison.png`, and `baseline_method_metadata.csv`. The metadata file is the code-side record for the comparison table:

| Method | Project configuration | Reference |
| --- | --- | --- |
| DP-FedAvg (`dp_fedavg`) | Client-side DP FedAvg | McMahan et al., AISTATS 2017; McMahan et al., ICLR 2018. |
| DP-FedProx (`dp_fedprox`) | Client-side DP update plus FedProx proximal term | Li et al., *Federated optimization in heterogeneous networks*, MLSys, 2020. |
| DP-FedNova (`dp_fednova`) | Client-side DP with normalized aggregation for heterogeneous local steps | Wang et al., *Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization*, NeurIPS, 2020. |
| DP-FedAdam (`dp_fedadam`) | Client-side DP with server-side adaptive FedOpt aggregation | Reddi et al., *Adaptive Federated Optimization*, ICLR, 2021. |
| GRAIL-FL (`grail_fl`) | Client-side DP plus regulated resource-privacy-governance orchestration | Proposed method; must enable `regulated_generic`. |

Legacy baselines remain available for reproducibility checks, but the final main table should use only the five methods above.

### Additional CUDA Suites

Participation-policy comparison:

```powershell
python APDP-RTFL/main.py --experiment-suite participation --participation-policies all,random,apdp_score --participation-rate 0.6 --run-name participation_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

Privacy-budget sensitivity:

```powershell
python APDP-RTFL/main.py --experiment-suite privacy_sensitivity --privacy-budgets 20,50,80,100 --privacy-sensitivity-methods dp_fedavg,dp_fedprox,dp_fednova,dp_fedadam,grail_fl --run-name privacy_sensitivity_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

Client-level fairness:

```powershell
python APDP-RTFL/main.py --experiment-suite fairness --fairness-methods dp_fedavg,dp_fedprox,dp_fednova,dp_fedadam,grail_fl --run-name fairness_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

## 2. Regulatory Intervention

Use this experiment to measure warning, downweighting, quarantine, and their impact on utility. Keep the command identical to the DP baseline except for the intervention switch.

```powershell
python APDP-RTFL/main.py --experiment-suite baselines --methods grail_fl --enable-regulatory-intervention --reg-warning-threshold 1.5 --reg-quarantine-threshold 2.5 --reg-penalty-weight 0.5 --run-name regulatory_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

Compare it with the same command without `--enable-regulatory-intervention`. Main outputs: `regulatory_intervention_summary.csv`, `regulatory_actions.png`, and `regulatory_risk_by_client.png`.

## 3. Pollution Detection And Regulatory Intervention

Pollution is disabled in ordinary runs. It is enabled only by `--experiment-suite pollution` together with `--enable-pollution-injection`.

Label-flipping scenario:

```powershell
python APDP-RTFL/main.py --experiment-suite pollution --methods grail_fl --enable-pollution-injection --pollution-type label_flip --polluted-clients 1,3 --pollution-start-round 10 --enable-regulatory-intervention --run-name pollution_label_flip_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

Feature-noise scenario:

```powershell
python APDP-RTFL/main.py --experiment-suite pollution --methods grail_fl --enable-pollution-injection --pollution-type feature_noise --polluted-clients 1,3 --pollution-start-round 10 --enable-regulatory-intervention --run-name pollution_feature_noise_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

Report the utility change, detection rate, false positive rate, and false negative rate from `pollution_final_metrics.csv`, `pollution_summary.csv`, `pollution_injection_summary.csv`, and `pollution_detection_rate.png`.

## 4. Synthetic Sensitive-Attribute Fairness Stress Test

This is a client-level synthetic stress test, not an evaluation of real demographic fairness. It creates linked differences in sample coverage, label distribution, feature quality, participation stability, and compute capability.

```powershell
python APDP-RTFL/main.py --experiment-suite synthetic_fairness --fairness-datasets emnist --fairness-methods dp_fedavg,dp_fedprox,dp_fednova,dp_fedadam,grail_fl --synthetic-sensitive-attrs gender,age,region --fairness-pressure-profile regulated --run-name synthetic_fairness_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

For FEMNIST, CIFAR10, and CIFAR100, first generate `data/<dataset>/all_data`, then replace `--fairness-datasets emnist` with the requested comma-separated datasets. Main outputs: `synthetic_sensitive_clients.csv`, `synthetic_group_fairness_summary.csv`, `federated_group_fairness_summary.csv`, and the group fairness charts.

Suggested manuscript wording: this study uses client-level synthetic sensitive attributes to simulate institutional heterogeneity and resource imbalance. They do not represent real gender, age, or regional demographic attributes.

## 5. Penalty And Approximate Shapley Contribution

The contribution evaluator uses leave-one-out marginal utility as an approximate Shapley value. In the manuscript, call it an approximate Shapley value or leave-one-out marginal contribution, not an exact Shapley computation.

```powershell
python APDP-RTFL/main.py --experiment-suite contribution --contribution-methods dp_fedavg,dp_fedprox,dp_fednova,dp_fedadam,grail_fl --contribution-quality-weight 0.25 --contribution-shapley-weight 0.35 --contribution-risk-weight 0.30 --contribution-fairness-weight 0.10 --contribution-utility-metric balanced_accuracy --enable-regulatory-intervention --run-name contribution_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

Use `contribution_penalty_summary.csv`, `approx_shapley_by_client.png`, `penalty_components.png`, and `contribution_weight_alignment.png` to discuss the balance between data quality, regulatory-risk penalties, fairness penalties, and aggregation weights.

## 6. Audit Traceability

This suite automatically records a client-by-round SHA-256 hash chain and records regulatory, fairness, and contribution observations during the normal GRAIL-FL training process.

```powershell
python APDP-RTFL/main.py --experiment-suite audit_trace --audit-methods dp_fedavg,dp_fedprox,dp_fednova,dp_fedadam,grail_fl --audit-digest-algorithm sha256 --run-name audit_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

Report `audit_trace_summary.csv`, `audit_chain_verification.csv`, `audit_trace_log.csv`, and `audit_trace_timeline.png`. The verification table should show zero invalid chain links in a successful run.

## 7. Component Ablation

Run all scenarios with the same seed set and report the absolute and relative change against `full`.

```powershell
python APDP-RTFL/main.py --experiment-suite ablation --ablation-method grail_fl --ablation-scenarios full,no_adaptive_privacy,no_compute_adapter,no_resource_orchestration,no_partial_updates,no_resource_fairness,no_opportunity_privacy,no_budget_utilization_boost,no_low_resource_compensation,no_zkip,no_ebcd,no_tcm,no_regulatory,no_contribution,no_fairness --run-name ablation_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

Main outputs: `ablation_final_metrics.csv`, `ablation_summary.csv`, `ablation_accuracy.png`, `ablation_macro_f1.png`, `ablation_balanced_accuracy.png`, and `ablation_accuracy_delta.png`.

GRAIL-FL-related ablation interpretation:

| Scenario | Removed component | Primary observations |
| --- | --- | --- |
| `no_resource_orchestration` | Resource-privacy joint orchestration path | Balanced accuracy, deadline success rate, slow-client participation |
| `no_partial_updates` | Force full upload and remove parameter-block partial updates | Uploaded bytes, deadline success rate, accuracy |
| `no_resource_fairness` | Remove resource-tier rotating coverage constraint | Participation gaps across constrained/standard/high tiers |
| `no_opportunity_privacy` | Remove future-opportunity-aware privacy spending | Epsilon utilization, low-resource-tier noise multiplier, accuracy |
| `no_budget_utilization_boost` | Remove closed-loop budget-utilization boost | Final epsilon utilization, convergence speed, accuracy |
| `no_low_resource_compensation` | Remove privacy-spend compensation for clients with scarce future participation | Constrained-tier epsilon utilization and effective participation rate |

Do not report ablation only by final accuracy. Also report tier-level epsilon utilization, effective participation rate, upload ratio, and residual-feedback count from `tier_privacy_summary.csv`, so the reader can see whether performance changes come from bringing low-resource clients back into effective training.

## Recommended Paper Result Matrix

| Paper question | Primary suite | Minimum reporting |
| --- | --- | --- |
| DP utility and privacy trade-off | `baselines` | Accuracy, Macro-F1, Balanced Accuracy, AUC when defined, noise scale, runtime |
| Resource-privacy joint orchestration | `baselines` + `heterogeneity_profile=regulated_generic` | Tier epsilon utilization, deadline success rate, upload ratio, residual-feedback count, slow-client effective participation |
| Timely regulatory intervention | `baselines` with intervention | Warnings, downweighting, quarantine, utility change |
| Pollution detection | `pollution` | Detection rate, false positives, false negatives, utility under attack |
| Synthetic group fairness | `synthetic_fairness` | Worst-group accuracy, accuracy/F1 gaps, epsilon gap, participation gap |
| Quality and contribution governance | `contribution` | Approximate Shapley, risk/fairness penalties, contribution-weight alignment |
| Traceability | `audit_trace` | Audit events, verified links, invalid links, per-round audit fields |
| Mechanism necessity | `ablation` | Full-versus-removed-component performance deltas |

Before pooling seeds, retain each raw run directory. Aggregate only the relevant final-metric CSV files into a separate analysis table; never overwrite raw audit, regulatory, or pollution logs.

## 8. Multi-Seed Aggregation And Paper Tables

Run each formal command once per seed, keeping a stable prefix pattern. For the new GRAIL-FL main baseline, use `--run-name grail_main_emnist_seed42`, `grail_main_emnist_seed43`, and `grail_main_emnist_seed44`; the runner produces directories such as `grail_main_emnist_seed42_20260625_123417` automatically. Do not reuse old APDP result prefixes when preparing the final paper table.

Aggregate raw baseline directories without modifying them:

```powershell
python APDP-RTFL/aggregate_results.py --input-root results --run-pattern grail_main_emnist_seed* --input-file baseline_final_metrics.csv --output-dir results/grail_main_emnist_aggregate --title-prefix "EMNIST DP Baselines with GRAIL-FL"
```

The aggregator creates three CSV files:

| Output file | Use |
| --- | --- |
| `experiment_seed_metrics.csv` | One method row per raw run, including the seed inferred from its directory name. Use for traceability and statistical checks. |
| `experiment_metric_summary.csv` | Long-form `method-metric-mean-std-n` summary for plotting or supplementary material. |
| `experiment_paper_main_table.csv` | Wide-format table with `<metric>_mean`, `<metric>_std`, and `<metric>_n` columns. Use as the source for the primary results table. |

It also creates `aggregate_<metric>.png` files with mean plus sample-standard-deviation error bars. For another suite, point `--input-file` at its own final metrics file, for example `pollution_final_metrics.csv`, `ablation_final_metrics.csv`, or `privacy_sensitivity_final_metrics.csv`, and use a separate aggregate output directory.

GRAIL-FL resource-tier diagnostics require a separate aggregation pass. The script reads each run directory's `grail_fl/tier_privacy_summary.csv` and `grail_fl/resource_privacy_diagnostics.csv`; if the input is an ablation directory, it also recursively reads diagnostics under `scenario/grail_fl/`:

```powershell
python APDP-RTFL/aggregate_arpa_diagnostics.py --input-root results --run-pattern grail_main_emnist_seed* --output-dir results/grail_emnist_diagnostics_aggregate --title-prefix "EMNIST GRAIL-FL Diagnostics"
```

For formal acceptance, add `--require-complete` so any run missing `tier_privacy_summary.csv` fails the aggregation instead of being silently skipped:

```powershell
python APDP-RTFL/aggregate_arpa_diagnostics.py --input-root results --run-pattern grail_main_emnist_seed* --require-complete --output-dir results/grail_emnist_diagnostics_aggregate_strict --title-prefix "EMNIST GRAIL-FL Diagnostics"
```

Main outputs:

| Output file | Use |
| --- | --- |
| `arpa_tier_seed_metrics.csv` | One row per raw run and resource tier, retaining seed, scenario, method, and tier. |
| `arpa_tier_metric_summary.csv` | Long-form resource-tier metric means, standard deviations, and sample sizes. |
| `arpa_tier_paper_table.csv` | Wide-format table that can be directly organized into a mechanism-diagnostics table. |
| `arpa_client_seed_diagnostics.csv` | Merged client-level resource-privacy diagnostics. |
| `aggregate_tier_<metric>_<scenario>.png` | Mean and standard-deviation charts for each resource-tier metric. |

Default diagnostics include `avg_epsilon_utilization`, `avg_historical_success_rate`, `avg_deadline_feasible_rate`, `avg_noise_multiplier`, `avg_upload_ratio`, `avg_residual_pressure`, `compressed_selection_count`, and `residual_feedback_full_upload_count`. To focus on the core governance metrics in the main text, specify metrics explicitly:

```powershell
python APDP-RTFL/aggregate_arpa_diagnostics.py --input-root results --run-pattern grail_main_emnist_seed* --metrics avg_epsilon_utilization,avg_historical_success_rate,avg_upload_ratio,residual_feedback_full_upload_count --output-dir results/grail_emnist_diagnostics_core --title-prefix "EMNIST GRAIL-FL Core Diagnostics"
```

To focus only on the low-resource tier, use `--tiers constrained`. For ablation input, use `--scenario-filter full,no_partial_updates` to summarize only selected scenarios:

```powershell
python APDP-RTFL/aggregate_arpa_diagnostics.py --input-root results --run-pattern ablation_emnist_seed*_grail* --scenario-filter full,no_partial_updates,no_opportunity_privacy --tiers constrained --output-dir results/grail_ablation_constrained_diagnostics --title-prefix "Constrained-tier GRAIL-FL Ablation"
```

Default aggregation metrics are final Accuracy, Macro-F1, Balanced Accuracy, AUC, average round time, and average DP noise scale. When a table has a narrower purpose, specify the required metrics:

```powershell
python APDP-RTFL/aggregate_results.py --input-root results --run-pattern pollution_label_flip_seed*_grail* --input-file pollution_final_metrics.csv --metrics final_accuracy,final_f1_score,detection_rate,false_positive_rate,false_negative_rate --output-dir results/pollution_label_flip_grail_aggregate --title-prefix "Label-Flipping Detection with GRAIL-FL"
```

## 9. GRAIL-FL Single-Run Acceptance Check

After a method comparison containing GRAIL-FL and DP baselines finishes, use the acceptance script to check whether the run meets the pre-specified conditions. The script does not change raw training artifacts; it writes check tables into an independent output directory.

```powershell
python APDP-RTFL/validate_arpa_acceptance.py --run-dir results/grail_main_emnist_seed42_YYYYmmdd_HHMMSS --metric final_balanced_accuracy --baselines dp_fedavg,dp_fedprox,dp_fednova,dp_fedadam --require-all-baselines --min-win-margin 0.0 --min-epsilon-utilization 0.70 --min-constrained-success-rate 0.50 --max-deadline-failure-rate 0.20 --output-dir results/grail_main_emnist_seed42_acceptance
```

Acceptance checks:

| Check | Meaning |
| --- | --- |
| `requested_baselines_present` | Whether all requested baselines are present; formal acceptance should use `--require-all-baselines`. |
| `beats_present_baselines` | Whether GRAIL-FL beats the DP baselines present in this run on the selected primary metric. |
| `avg_tier_epsilon_utilization` | Whether average epsilon utilization across resource tiers meets the threshold. |
| `constrained_effective_participation` | Whether constrained-tier effective participation meets the threshold. |
| `predicted_deadline_failure_rate` | Whether selected clients' predicted deadline failure rate is below the threshold. |

Outputs include `arpa_acceptance_checks.csv`, `arpa_baseline_comparisons.csv`, and `arpa_acceptance_summary.json`. The comparison table records GRAIL-FL's metric difference against each baseline as `margin`; set `--min-win-margin` when GRAIL-FL should exceed baselines by a minimum margin. For formal acceptance, add `--strict` to require `grail_fl/tier_privacy_summary.csv` and `grail_fl/resource_trace.csv`.

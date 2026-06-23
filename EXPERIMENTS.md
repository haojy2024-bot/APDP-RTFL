# APDP-RTFL Experiment Command Guide

This guide defines reproducible commands for the regulated-industry APDP-RTFL study. Run every command from the repository root:

```powershell
python APDP-RTFL/main.py <arguments>
```

The result location is `results/<run-name>_YYYYmmdd_HHMMSS/`. Set `--run-name` explicitly as a logical experiment-name prefix for every formal run; the runner always appends its launch timestamp, including when a prefix is supplied. Use the same dataset split, client count, round count, partition, privacy-budget parameters, seed list, and backend within one comparison table or figure.

Each timestamped run directory is write-once: an existing final directory is rejected to prevent mixed artifacts. Every new run writes `run_config.json`, `run_command.txt`, `environment.json`, `data_artifacts/`, and `artifact_manifest.csv`. The data-artifact directory contains dataset fingerprints, client split summaries, and the generated failure plan; method directories with TCM enabled also contain `tcm_manifest.csv` and recoverable `checkpoints/*.npz` files.

## Common Formal Setting

The following is a starting protocol for a formal EMNIST experiment. It intentionally uses a non-IID partition and a fixed random seed. Repeat each formal command with at least three seeds, such as `42`, `43`, and `44`, and report mean plus standard deviation.

```powershell
--dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --min-epsilon 0.1 --max-epsilon 2 --dp-epsilon 1 --dp-delta 1e-5 --dp-l2-norm-clip 1 --backend sklearn --seed 42
```

For small smoke tests only, add `--max-samples 500 --num-rounds 2 --num-clients 5`. Do not use these small-sample results in the paper.

## 1. DP Baseline Comparison

### Client-side DP-SGD accounting

Client-DP methods use sample-level local DP-SGD: clipping and Gaussian noise are applied locally to per-sample gradients, while the server does not add new aggregation-side DP noise. `--epsilon-per-client-total` is the target `(epsilon, delta)` budget for each client's complete training trace (default epsilon 5, delta `1e-5`), accumulated with an RDP accountant from actual DP-SGD steps. The legacy `--total-privacy-budget` option is retained only as a compatibility alias and is no longer a budget pool divided among clients or reset each round.

The expected local DP-SGD batch size is 256 (`--dp-batch-size`). Client-DP outputs include `privacy_accounting.csv` and `privacy_accounting_summary.csv`; report final client epsilon, target epsilon, and any budget-exhausted events.

The default `--methods all` contains only the DP comparison set:

`DP-FL`, `DP-FLProx`, `DP-FedSGD`, `DP-RTFL`, and `APDP-RTFL`. `Global-DP` remains available only when explicitly requested and is not part of the default main table.

```powershell
python APDP-RTFL/main.py --experiment-suite baselines --methods all --run-name baseline_emnist_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --min-epsilon 0.1 --max-epsilon 2 --dp-epsilon 1 --dp-delta 1e-5 --dp-l2-norm-clip 1 --backend sklearn --seed 42
```

Key outputs are `baseline_final_metrics.csv`, `baseline_summary.csv`, `baseline_comparison.png`, and `baseline_method_metadata.csv`. The metadata file is the code-side record for the comparison table:

| Method | Project configuration | Reference |
| --- | --- | --- |
| DP-FL | Client-side DP update | Arachchige et al., *Local differential privacy for deep learning*, IEEE IoT Journal, 2019/2020. |
| DP-FLProx | Client-side DP update plus FedProx proximal term | Li et al., *Federated optimization in heterogeneous networks*, MLSys, 2020. |
| DP-FedSGD | Client-side DP update with one forced local epoch | Auddy et al., *Statistical Limits and Efficient Algorithms for Differentially Private Federated Learning*, arXiv:2605.18656, 2026. |
| Global-DP | Server-side noise after aggregation | Project implementation baseline. |
| DP-RTFL | DP plus ZKIP, EBCD, and TCM | Project implementation baseline. |
| APDP-RTFL | DP-RTFL plus adaptive privacy and compute adaptation | Proposed method. |

`DP-FedSGD` is a controlled project configuration, not a claim of exact reproduction of the cited work. `FedAvg`, `FedProx`, and `LDP-FL` remain available only when explicitly named and should not appear in the primary DP-only table.

## 2. Regulatory Intervention

Use this experiment to measure warning, downweighting, quarantine, and their impact on utility. Keep the baseline command identical except for the intervention switch.

```powershell
python APDP-RTFL/main.py --experiment-suite baselines --methods apdp_rtfl --enable-regulatory-intervention --reg-warning-threshold 1.5 --reg-quarantine-threshold 2.5 --reg-penalty-weight 0.5 --run-name regulatory_emnist_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --backend sklearn --seed 42
```

Compare it with the same command without `--enable-regulatory-intervention`. Main outputs: `regulatory_intervention_summary.csv`, `regulatory_actions.png`, and `regulatory_risk_by_client.png`.

## 3. Pollution Detection and Intervention

Pollution is disabled in ordinary runs. It is enabled only by `--experiment-suite pollution` together with `--enable-pollution-injection`.

Label-flipping scenario:

```powershell
python APDP-RTFL/main.py --experiment-suite pollution --methods apdp_rtfl --enable-pollution-injection --pollution-type label_flip --polluted-clients 1,3 --pollution-start-round 10 --enable-regulatory-intervention --run-name pollution_label_flip_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --backend sklearn --seed 42
```

Feature-noise scenario:

```powershell
python APDP-RTFL/main.py --experiment-suite pollution --methods apdp_rtfl --enable-pollution-injection --pollution-type feature_noise --polluted-clients 1,3 --pollution-start-round 10 --enable-regulatory-intervention --run-name pollution_feature_noise_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --backend sklearn --seed 42
```

Report the utility change, detection rate, false positive and false negative rates from `pollution_final_metrics.csv`, `pollution_summary.csv`, `pollution_injection_summary.csv`, and `pollution_detection_rate.png`.

## 4. Synthetic Sensitive-Attribute Fairness Stress Test

This is a client-level synthetic stress test, not an evaluation of real demographic fairness. It creates linked differences in sample coverage, label distribution, feature quality, availability, and compute capability.

```powershell
python APDP-RTFL/main.py --experiment-suite synthetic_fairness --fairness-datasets emnist --fairness-methods dp_fl,dp_flprox,dp_fedsgd,global_dp,dp_rtfl,apdp_rtfl --synthetic-sensitive-attrs gender,age,region --fairness-pressure-profile regulated --run-name synthetic_fairness_emnist_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --backend sklearn --seed 42
```

For FEMNIST, CIFAR10, and CIFAR100, replace `--fairness-datasets emnist` with the requested comma-separated datasets after `data/<dataset>/all_data` has been generated. Main outputs: `synthetic_sensitive_clients.csv`, `synthetic_group_fairness_summary.csv`, `federated_group_fairness_summary.csv`, and the group fairness charts.

Suggested manuscript wording: the attributes are synthetic client-level proxies for institutional and resource heterogeneity. They do not represent real gender, age, or regional demographics.

## 5. Penalty and Approximate Shapley Contribution

The contribution evaluator uses leave-one-out marginal utility as an approximate Shapley value. Do not call it an exact Shapley computation in the manuscript.

```powershell
python APDP-RTFL/main.py --experiment-suite contribution --contribution-methods dp_fl,dp_rtfl,apdp_rtfl --contribution-quality-weight 0.25 --contribution-shapley-weight 0.35 --contribution-risk-weight 0.30 --contribution-fairness-weight 0.10 --contribution-utility-metric balanced_accuracy --enable-regulatory-intervention --run-name contribution_emnist_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --backend sklearn --seed 42
```

Use `contribution_penalty_summary.csv`, `approx_shapley_by_client.png`, `penalty_components.png`, and `contribution_weight_alignment.png` to discuss the balance between data quality, risk penalties, fairness penalties, and aggregation weight.

## 6. Audit Traceability

This suite automatically records a client-by-round SHA-256 hash chain and evaluates the normal APDP-RTFL process with regulatory, fairness, and contribution observations.

```powershell
python APDP-RTFL/main.py --experiment-suite audit_trace --audit-methods apdp_rtfl --audit-digest-algorithm sha256 --run-name audit_emnist_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --backend sklearn --seed 42
```

Report `audit_trace_summary.csv`, `audit_chain_verification.csv`, `audit_trace_log.csv`, and `audit_trace_timeline.png`. The verification table should show zero invalid chain links in a successful run.

## 7. Component Ablation

Run all scenarios with the same seed set and report the absolute and relative change against `full`.

```powershell
python APDP-RTFL/main.py --experiment-suite ablation --ablation-method apdp_rtfl --ablation-scenarios full,no_adaptive_privacy,no_compute_adapter,no_zkip,no_ebcd,no_tcm,no_regulatory,no_contribution,no_fairness --run-name ablation_emnist_seed42 --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 50 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --total-privacy-budget 5 --backend sklearn --seed 42
```

Main outputs: `ablation_final_metrics.csv`, `ablation_summary.csv`, `ablation_accuracy.png`, `ablation_macro_f1.png`, `ablation_balanced_accuracy.png`, and `ablation_accuracy_delta.png`.

## Recommended Paper Result Matrix

| Paper question | Primary suite | Minimum reporting |
| --- | --- | --- |
| DP utility and privacy trade-off | `baselines` | Accuracy, macro-F1, balanced accuracy, AUC where defined, noise scale, runtime |
| Timely regulatory intervention | `baselines` with intervention | Warnings, downweighting, quarantine, utility change |
| Pollution detection | `pollution` | Detection rate, false positives, false negatives, utility under attack |
| Synthetic group fairness | `synthetic_fairness` | Worst-group accuracy, accuracy/F1 gaps, epsilon and participation gaps |
| Quality and contribution governance | `contribution` | Approximate Shapley, risk/fairness penalties, contribution-weight alignment |
| Traceability | `audit_trace` | Audit events, verified links, invalid links, per-round trace fields |
| Mechanism necessity | `ablation` | Full-versus-removed-component performance deltas |

Before pooling seeds, retain each raw run directory. Aggregate only the relevant final-metric CSV files into a separate analysis table; never overwrite the raw audit or intervention logs.

## 8. Multi-Seed Aggregation and Paper Tables

Run each formal command once per seed, keeping a stable prefix pattern. For example, use `--run-name baseline_emnist_seed42`, `baseline_emnist_seed43`, and `baseline_emnist_seed44`; the runner produces directories such as `baseline_emnist_seed42_20260623_103617` automatically.

Aggregate the raw baseline directories without modifying them:

```powershell
python APDP-RTFL/aggregate_results.py --input-root results --run-pattern baseline_emnist_seed* --input-file baseline_final_metrics.csv --output-dir results/baseline_emnist_aggregate --title-prefix "EMNIST DP Baselines"
```

The aggregator creates three CSV files:

| Output | Use |
| --- | --- |
| `experiment_seed_metrics.csv` | One method row per raw run, including the seed inferred from its directory name. Use for traceability and statistical checks. |
| `experiment_metric_summary.csv` | Long-form `method-metric-mean-std-n` summary for plotting or supplementary material. |
| `experiment_paper_main_table.csv` | Wide-format table with `<metric>_mean`, `<metric>_std`, and `<metric>_n` columns. Use as the source for the primary results table. |

It also creates `aggregate_<metric>.png` files with mean plus sample-standard-deviation error bars. For another suite, point `--input-file` at its own final metrics file, for example `pollution_final_metrics.csv`, `ablation_final_metrics.csv`, or `privacy_sensitivity_final_metrics.csv`, and use a separate aggregate output directory.

The default metrics are final accuracy, macro-F1, balanced accuracy, AUC, average round time, and average DP noise scale. Override them when a table has a narrower purpose:

```powershell
python APDP-RTFL/aggregate_results.py --input-root results --run-pattern pollution_label_flip_seed* --input-file pollution_final_metrics.csv --metrics final_accuracy,final_f1_score,detection_rate,false_positive_rate,false_negative_rate --output-dir results/pollution_label_flip_aggregate --title-prefix "Label-Flipping Detection"
```

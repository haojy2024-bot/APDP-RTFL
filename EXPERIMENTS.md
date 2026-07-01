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

`DP-FedAvg`, `DP-FedProx`, `DP-FedSGD`, `DP-FedNova`, and the proposed `GRAIL-FL`. Legacy methods such as `dp_fl`, `dp_flprox`, `ldp_fl`, `global_dp`, `dp_rtfl`, `apdp_rtfl`, and `dp_fedadam` remain explicitly runnable for older-result audits, but they should not be placed in the final main table.

## Client-Side DP-SGD And Privacy Accounting

Client-DP methods use sample-level local DP-SGD: each client runs Poisson mini-batches locally with per-sample gradient clipping and Gaussian noise, while the server does not add new aggregation-side DP noise. `--epsilon-per-client-total` is the target budget for each client's complete training trace, with default epsilon 5 and default `--dp-delta 1e-5`; the RDP accountant accumulates the actual DP-SGD steps. The old `--total-privacy-budget` option is only a compatibility alias for this parameter and no longer means a shared budget divided between clients or reset every round.

The expected local DP-SGD batch size is 256 and can be changed through `--dp-batch-size`. Every client-DP run writes `privacy_accounting.csv` and `privacy_accounting_summary.csv`; report each client's final epsilon, target epsilon, and number of budget-exhausted events.

## PyTorch/GPU Backend

Use `--backend torch --device cuda` to run the client model, local client training tensors, per-sample gradient clipping, and Gaussian DP-SGD noise generation on the GPU. The default `--torch-model linear` preserves the historical linear softmax/logistic model; use `--torch-model mlp --torch-mlp-hidden 256,128` to diagnose model-capacity limits with a small fully connected network. When combined with `--heterogeneity-profile regulated_generic`, torch uses the same full GRAIL-FL runner as sklearn across the paper experiment suites: resource orchestration, client selection, privacy spending, partial upload, RDP accounting, ZKIP/EBCD/TCM, regulatory intervention, contribution scoring, audit traceability, and mechanism diagnostics keep the same semantics. For CUDA paper runs, use `--dp-batch-size 256 --torch-batch-size 256` together with the regulated resource profile.

Install a CUDA-enabled PyTorch build on the experiment server before using this backend, then verify the device:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

The command should print `True`.

### S2: Model-Capacity And Privacy-Strength Diagnostics

Before continuing formal main-table experiments, run diagnostics with the same data partition, client count, and model backbone. This stage is not the paper main table; it determines whether the current `0.5-0.6` accuracy is primarily caused by model capacity, DP noise, or GRAIL-FL scheduling. This paper does not force a fixed `epsilon=5` final privacy budget; privacy is an adjustable experimental variable. The formal main table should use a calibrated unified budget that converges reliably, and the table header and text must report that budget explicitly. To save compute, S2 diagnostics use one random seed only, defaulting to `seed=42`; expand to `42/43/44` only after a configuration is promising enough for the formal main table. Start with EMNIST balanced. If FEMNIST remains abnormal, diagnose FEMNIST separately with reduced task difficulty.

Diagnostic naming:

| Diagnostic | Purpose | Methods | Privacy budget | Recommended prefix |
| --- | --- | --- | --- | --- |
| no-DP upper bound | Estimate the MLP upper bound under the current FL split | `fedavg,fedprox` | no client DP | `s2_upper_${dataset}_seed42` |
| weak DP | Check whether relaxed DP approaches the no-DP upper bound | `dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl` | `epsilon_per_client_total=20` | `s2_weakdp_${dataset}_seed42` |
| tight-budget reference DP | Estimate the lower-bound performance under a stricter budget; not a mandatory main-table budget | `dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl` | `epsilon_per_client_total=5` | `s2_tightdp_${dataset}_seed42` |

No-DP upper-bound diagnostic:

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

Weak-DP diagnostic:

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

Tight-budget reference DP diagnostic:

```bash
python APDP-RTFL/main.py \
  --experiment-suite baselines \
  --methods dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl \
  --run-name s2_tightdp_emnist_seed42 \
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

Interpretation rules:

- If the no-DP upper bound remains below 0.70, the main bottleneck is model capacity, data partitioning, or training intensity rather than DP.
- If no-DP is clearly above 0.70 and weak DP is close to no-DP, the model capacity is adequate and the formal-budget gap is mainly a DP-noise effect.
- If weak DP is much better than tight-budget reference DP, the model is privacy-budget sensitive; choose a unified budget that guarantees convergence for the formal main table, then present stricter budgets as sensitivity analysis.
- If GRAIL-FL is below fixed DP baselines in both weak and tight-budget reference DP, reduce `arpa-privacy-boost-gain` and `arpa-max-privacy-boost` so budget utilization gains are not canceled by larger noise.
- S2 diagnostics are only for choosing the formal configuration. The final main table must still use the same backbone, calibrated privacy budget, data split, and seed set across methods.

### S2.5: Weak-DP Trainability Tuning

If S2 shows that the no-DP upper bound is already above `0.70` but weak DP with `epsilon=20` remains below `0.60`, do not move directly to the formal main table and do not expand to multiple random seeds. S2.5 uses one random seed only, `seed=42`, and keeps the dataset, client count, partition, and model backbone fixed. Its purpose is to find the privacy budget, clipping threshold, and local training intensity that converge reliably.

Run S2.5 in this order:

| Order | Diagnostic | Methods | Key change | Move on when |
| --- | --- | --- | --- | --- |
| A | Complete missing weak-DP methods | `dp_fedsgd`, `dp_fednova` | Same as S2 weak DP, one method per run | Each method has a full 200-round result |
| B | Longer convergence check | `dp_fedavg`, `grail_fl` | `num_rounds=400`, other parameters unchanged | Last-50-round gain is below `0.01` or accuracy exceeds `0.65` |
| C | DP-SGD mini-grid | First `dp_fedavg`, then `grail_fl` | Tune `dp_batch_size`, `dp_l2_norm_clip`, and `client_epochs` | At least one configuration reaches `0.65` |
| D | GRAIL-FL scheduling trigger | `grail_fl` | Lower deadline or raise reference batch to trigger partial upload | `compressed_selection_count > 0` and accuracy is not below fixed DP baselines |

Execution rules:

- Every command still uses only `--seed 42`.
- Run one method or one configuration at a time so an interruption does not invalidate a whole batch.
- If `dp_fedavg` remains below `0.60` in S2.5, do not continue to the formal main table; inspect DP-SGD implementation, gradient clipping, and noise logging first, or relax the privacy budget further.
- If `dp_fedavg` reaches `0.65-0.70`, rerun `grail_fl` with the same configuration.
- If `grail_fl` has similar accuracy to fixed DP baselines but better epsilon utilization, deadline behavior, upload ratio, or fairness metrics, proceed to candidate main-table reruns.

#### S2.5-A: Complete Missing Weak-DP Methods

First run `dp_fedsgd`:

```bash
python APDP-RTFL/main.py \
  --experiment-suite baselines \
  --methods dp_fedsgd \
  --run-name s2_5_weakdp_dpfedsgd_emnist_seed42 \
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

Then run `dp_fednova`:

```bash
python APDP-RTFL/main.py \
  --experiment-suite baselines \
  --methods dp_fednova \
  --run-name s2_5_weakdp_dpfednova_emnist_seed42 \
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

#### S2.5-B: Longer Weak-DP Convergence Check

If the last 20 rounds in S2 are still improving, use `num_rounds=400` to check whether weak DP is simply converging slowly. Run `dp_fedavg` first; if it exceeds `0.65`, rerun `grail_fl`.

```bash
python APDP-RTFL/main.py \
  --experiment-suite baselines \
  --methods dp_fedavg \
  --run-name s2_5_weakdp_dpfedavg_r400_emnist_seed42 \
  --dataset emnist \
  --emnist-split balanced \
  --num-clients 20 \
  --num-rounds 400 \
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
  --failure-prob 0 \
  --seed 42
```

#### S2.5-C: DP-SGD Mini-Grid

If 400 rounds remain below `0.65`, run the mini-grid with `dp_fedavg` first to avoid mixing in GRAIL-FL mechanism effects. Run the rows one at a time:

| Config | `num_rounds` | `client_epochs` | `dp_batch_size` | `torch_batch_size` | `dp_l2_norm_clip` | Purpose |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `batch512_clip1_e3` | 200 | 3 | 512 | 512 | 1.0 | Reduce batch-averaged noise impact |
| `batch512_clip2_e3` | 200 | 3 | 512 | 512 | 2.0 | Check whether clipping is too aggressive |
| `batch512_clip05_e3` | 200 | 3 | 512 | 512 | 0.5 | Check whether smaller clipping improves stability |
| `batch512_clip1_e5` | 200 | 5 | 512 | 512 | 1.0 | Increase local training intensity |
| `batch512_clip2_e5` | 200 | 5 | 512 | 512 | 2.0 | Relax clipping and increase local training together |

Command template, replacing only `run-name`, `client-epochs`, `dp-batch-size`, `torch-batch-size`, and `dp-l2-norm-clip`:

```bash
python APDP-RTFL/main.py \
  --experiment-suite baselines \
  --methods dp_fedavg \
  --run-name s2_5_weakdp_dpfedavg_batch512_clip1_e3_emnist_seed42 \
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
  --dp-batch-size 512 \
  --torch-batch-size 512 \
  --backend torch \
  --device cuda \
  --torch-model mlp \
  --torch-mlp-hidden 256,128 \
  --heterogeneity-profile regulated_generic \
  --failure-prob 0 \
  --seed 42
```

#### S2.5-D: GRAIL-FL Scheduling Trigger

After finding a stronger DP-SGD configuration in S2.5-C, rerun GRAIL-FL with the same configuration and lower the deadline so partial upload actually triggers. Start with a mild setting to avoid excessive compression:

```bash
python APDP-RTFL/main.py \
  --experiment-suite baselines \
  --methods grail_fl \
  --run-name s2_5_weakdp_grail_trigger_emnist_seed42 \
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
  --dp-batch-size 512 \
  --torch-batch-size 512 \
  --backend torch \
  --device cuda \
  --torch-model mlp \
  --torch-mlp-hidden 256,128 \
  --heterogeneity-profile regulated_generic \
  --round-deadline-seconds 1.5 \
  --reference-batch-seconds 0.02 \
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

S2.5 pass criteria:

- Weak-DP `dp_fedavg` or `grail_fl` final accuracy is above `0.65`, with macro-F1 improving as well.
- Average final epsilon is close to `20`, with `0` budget-exhausted events.
- If GRAIL-FL is included, it should have `compressed_selection_count > 0` or a clear mechanism advantage in resource/privacy-utilization metrics.
- Only after passing S2.5 should you run candidate main-table reruns; otherwise inspect DP-SGD noise implementation and clipping statistics first, or enter S2.5-C3 relaxed-privacy calibration.

#### S2.5-C3: 200-Round Relaxed-Privacy And Strong-Local-Training Calibration

If time is tight and the immediate goal is to reach about `0.70` accuracy within roughly 200 rounds, skip the 400-round grid and run a relaxed-privacy calibration. This stage still uses `seed=42`; first find a trainable `dp_fedavg` setting, then rerun `grail_fl` with the same setting. This is utility-oriented relaxed-DP calibration: the paper must report the actual `epsilon_per_client_total`. If this budget is selected for the formal main table, all main-table methods must use the same budget for a fair comparison.

Rules:

- Relax the total privacy budget first: start with `epsilon_per_client_total=50`; if accuracy is still below `0.68`, try `80`.
- Keep `num_rounds=200`; use `client_epochs=8` and `dp_l2_norm_clip=2/3` to strengthen local training.
- Prefer `dp_batch_size=256`, because S2.5-C showed that batch512 is unstable in the current implementation.
- Once `dp_fedavg` reaches `0.68-0.72`, stop further relaxation and rerun `grail_fl` with the same setting.

Priority command 1: `epsilon=50, clip=2, epochs=8`

```bash
python APDP-RTFL/main.py \
  --experiment-suite baselines \
  --methods dp_fedavg \
  --run-name s2_5_c3_relaxed_eps50_clip2_e8_dpfedavg_emnist_seed42 \
  --dataset emnist \
  --emnist-split balanced \
  --num-clients 20 \
  --num-rounds 200 \
  --client-epochs 8 \
  --partition dirichlet \
  --dirichlet-alpha 0.5 \
  --epsilon-per-client-total 50 \
  --min-epsilon 0.2 \
  --max-epsilon 10 \
  --dp-epsilon 1 \
  --dp-delta 1e-5 \
  --dp-l2-norm-clip 2 \
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

Priority command 2: `epsilon=50, clip=3, epochs=8`

```bash
python APDP-RTFL/main.py \
  --experiment-suite baselines \
  --methods dp_fedavg \
  --run-name s2_5_c3_relaxed_eps50_clip3_e8_dpfedavg_emnist_seed42 \
  --dataset emnist \
  --emnist-split balanced \
  --num-clients 20 \
  --num-rounds 200 \
  --client-epochs 8 \
  --partition dirichlet \
  --dirichlet-alpha 0.5 \
  --epsilon-per-client-total 50 \
  --min-epsilon 0.2 \
  --max-epsilon 10 \
  --dp-epsilon 1 \
  --dp-delta 1e-5 \
  --dp-l2-norm-clip 3 \
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

Priority command 3: `epsilon=80, clip=2, epochs=8`

```bash
python APDP-RTFL/main.py \
  --experiment-suite baselines \
  --methods dp_fedavg \
  --run-name s2_5_c3_relaxed_eps80_clip2_e8_dpfedavg_emnist_seed42 \
  --dataset emnist \
  --emnist-split balanced \
  --num-clients 20 \
  --num-rounds 200 \
  --client-epochs 8 \
  --partition dirichlet \
  --dirichlet-alpha 0.5 \
  --epsilon-per-client-total 80 \
  --min-epsilon 0.2 \
  --max-epsilon 15 \
  --dp-epsilon 1 \
  --dp-delta 1e-5 \
  --dp-l2-norm-clip 2 \
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

Priority command 4: if `dp_fedavg` reaches the target, rerun `grail_fl`

```bash
python APDP-RTFL/main.py \
  --experiment-suite baselines \
  --methods grail_fl \
  --run-name s2_5_c3_relaxed_eps50_clip2_e8_grail_emnist_seed42 \
  --dataset emnist \
  --emnist-split balanced \
  --num-clients 20 \
  --num-rounds 200 \
  --client-epochs 8 \
  --partition dirichlet \
  --dirichlet-alpha 0.5 \
  --epsilon-per-client-total 50 \
  --min-epsilon 0.2 \
  --max-epsilon 10 \
  --dp-epsilon 1 \
  --dp-delta 1e-5 \
  --dp-l2-norm-clip 2 \
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

Torch/GPU-GRAIL baseline commands for the four paper datasets:

Note: `--epsilon-per-client-total 5` below is a tight-budget reference value. If S2.5-C3 selects `50` or `80` as the formal utility-first budget, replace `--epsilon-per-client-total`, `--min-epsilon`, `--max-epsilon`, `--dp-l2-norm-clip`, and `--client-epochs` consistently across all formal baseline, participation, fairness, contribution, audit, and ablation commands.

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

CUDA is an experimental execution backend, not a claim that all real edge clients have GPUs. Resource heterogeneity remains controlled by the regulated resource profile and deadline simulation. The sklearn/GRAIL command below remains available as a CPU-compatible reference path for backend consistency checks.

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

Key outputs are `baseline_final_metrics.csv`, `baseline_summary.csv`, `baseline_comparison.png`, and `baseline_method_metadata.csv`. The metadata file is the code-side record for the comparison table:

| Method | Project configuration | Reference |
| --- | --- | --- |
| DP-FedAvg (`dp_fedavg`) | Client-side DP FedAvg | McMahan et al., AISTATS 2017; McMahan et al., ICLR 2018. |
| DP-FedProx (`dp_fedprox`) | Client-side DP update plus FedProx proximal term | Li et al., *Federated optimization in heterogeneous networks*, MLSys, 2020. |
| DP-FedSGD (`dp_fedsgd`) | Client-side DP with forced one-local-epoch FedSGD-style updates | McMahan et al., ICLR 2018; FedSGD-style one-step FL baseline. |
| DP-FedNova (`dp_fednova`) | Client-side DP with normalized aggregation for heterogeneous local steps | Wang et al., *Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization*, NeurIPS, 2020. |
| GRAIL-FL (`grail_fl`) | Client-side DP plus regulated resource-privacy-governance orchestration | Proposed method; must enable `regulated_generic`. |

Legacy baselines remain available for reproducibility checks, but the final main table should use only the five methods above.

### Additional CUDA Suites

Participation-policy comparison:

```powershell
python APDP-RTFL/main.py --experiment-suite participation --participation-policies all,random,apdp_score --participation-rate 0.6 --run-name participation_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

Privacy-budget sensitivity:

```powershell
python APDP-RTFL/main.py --experiment-suite privacy_sensitivity --privacy-budgets 20,50,80,100 --privacy-sensitivity-methods dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl --run-name privacy_sensitivity_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

Client-level fairness:

```powershell
python APDP-RTFL/main.py --experiment-suite fairness --fairness-methods dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl --run-name fairness_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
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
python APDP-RTFL/main.py --experiment-suite synthetic_fairness --fairness-datasets emnist --fairness-methods dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl --synthetic-sensitive-attrs gender,age,region --fairness-pressure-profile regulated --run-name synthetic_fairness_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

For FEMNIST, CIFAR10, and CIFAR100, first generate `data/<dataset>/all_data`, then replace `--fairness-datasets emnist` with the requested comma-separated datasets. Main outputs: `synthetic_sensitive_clients.csv`, `synthetic_group_fairness_summary.csv`, `federated_group_fairness_summary.csv`, and the group fairness charts.

Suggested manuscript wording: this study uses client-level synthetic sensitive attributes to simulate institutional heterogeneity and resource imbalance. They do not represent real gender, age, or regional demographic attributes.

## 5. Penalty And Approximate Shapley Contribution

The contribution evaluator uses leave-one-out marginal utility as an approximate Shapley value. In the manuscript, call it an approximate Shapley value or leave-one-out marginal contribution, not an exact Shapley computation.

```powershell
python APDP-RTFL/main.py --experiment-suite contribution --contribution-methods dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl --contribution-quality-weight 0.25 --contribution-shapley-weight 0.35 --contribution-risk-weight 0.30 --contribution-fairness-weight 0.10 --contribution-utility-metric balanced_accuracy --enable-regulatory-intervention --run-name contribution_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
```

Use `contribution_penalty_summary.csv`, `approx_shapley_by_client.png`, `penalty_components.png`, and `contribution_weight_alignment.png` to discuss the balance between data quality, regulatory-risk penalties, fairness penalties, and aggregation weights.

## 6. Audit Traceability

This suite automatically records a client-by-round SHA-256 hash chain and records regulatory, fairness, and contribution observations during the normal GRAIL-FL training process.

```powershell
python APDP-RTFL/main.py --experiment-suite audit_trace --audit-methods dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova,grail_fl --audit-digest-algorithm sha256 --run-name audit_emnist_seed42_grail --dataset emnist --emnist-split balanced --num-clients 20 --num-rounds 200 --client-epochs 3 --partition dirichlet --dirichlet-alpha 0.5 --epsilon-per-client-total 5 --dp-batch-size 256 --torch-batch-size 256 --backend torch --device cuda --heterogeneity-profile regulated_generic --round-deadline-seconds 5 --reference-batch-seconds 0.01 --parameter-blocks 8 --upload-ratios 1.0,0.5,0.25 --arpa-privacy-boost-gain 0.8 --arpa-max-privacy-boost 1.8 --arpa-opportunity-compensation-weight 0.65 --arpa-compression-slack-target 0.85 --arpa-residual-full-upload-threshold 0.25 --seed 42
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

Run each formal command once per dataset and seed, keeping a stable prefix pattern. For the new GRAIL-FL main baseline, use names such as `grail_main_emnist_seed42`, `grail_main_femnist_seed42`, `grail_main_cifar10_seed42`, and `grail_main_medmnist_seed42`; the runner appends timestamps automatically. Do not reuse old APDP result prefixes when preparing the final paper table.

Aggregate raw baseline directories without modifying them:

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

The aggregator creates three CSV files:

| Output file | Use |
| --- | --- |
| `experiment_seed_metrics.csv` | One method row per raw run, including the seed inferred from its directory name. Use for traceability and statistical checks. |
| `experiment_metric_summary.csv` | Long-form `method-metric-mean-std-n` summary for plotting or supplementary material. |
| `experiment_paper_main_table.csv` | Wide-format table with `<metric>_mean`, `<metric>_std`, and `<metric>_n` columns. Use as the source for the primary results table. |

It also creates `aggregate_<metric>.png` files with mean plus sample-standard-deviation error bars. For another suite, point `--input-file` at its own final metrics file, for example `pollution_final_metrics.csv`, `ablation_final_metrics.csv`, or `privacy_sensitivity_final_metrics.csv`, and use a separate aggregate output directory.

GRAIL-FL resource-tier diagnostics require a separate aggregation pass. The script reads each run directory's `grail_fl/tier_privacy_summary.csv` and `grail_fl/resource_privacy_diagnostics.csv`; if the input is an ablation directory, it also recursively reads diagnostics under `scenario/grail_fl/`:

```bash
for dataset in emnist femnist cifar10 medmnist; do
  python APDP-RTFL/aggregate_arpa_diagnostics.py \
    --input-root results \
    --run-pattern grail_main_${dataset}_seed* \
    --output-dir results/grail_${dataset}_diagnostics_aggregate \
    --title-prefix "${dataset} GRAIL-FL Diagnostics"
done
```

For formal acceptance, add `--require-complete` so any run missing `tier_privacy_summary.csv` fails the aggregation instead of being silently skipped:

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

Main outputs:

| Output file | Use |
| --- | --- |
| `arpa_tier_seed_metrics.csv` | One row per raw run and resource tier, retaining seed, scenario, method, and tier. |
| `arpa_tier_metric_summary.csv` | Long-form resource-tier metric means, standard deviations, and sample sizes. |
| `arpa_tier_paper_table.csv` | Wide-format table that can be directly organized into a mechanism-diagnostics table. |
| `arpa_client_seed_diagnostics.csv` | Merged client-level resource-privacy diagnostics. |
| `aggregate_tier_<metric>_<scenario>.png` | Mean and standard-deviation charts for each resource-tier metric. |

Default diagnostics include `avg_epsilon_utilization`, `avg_historical_success_rate`, `avg_deadline_feasible_rate`, `avg_noise_multiplier`, `avg_upload_ratio`, `avg_residual_pressure`, `compressed_selection_count`, and `residual_feedback_full_upload_count`. To focus on the core governance metrics in the main text, specify metrics explicitly:

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
python APDP-RTFL/validate_arpa_acceptance.py --run-dir results/grail_main_emnist_seed42_YYYYmmdd_HHMMSS --metric final_balanced_accuracy --baselines dp_fedavg,dp_fedprox,dp_fedsgd,dp_fednova --require-all-baselines --min-win-margin 0.0 --min-epsilon-utilization 0.70 --min-constrained-success-rate 0.50 --max-deadline-failure-rate 0.20 --output-dir results/grail_main_emnist_seed42_acceptance
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

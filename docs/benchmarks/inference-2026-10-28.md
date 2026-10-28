# Inference benchmark configuration — 2026-10-28

This records the settings behind the four measured configurations. For a runnable profile using
only options available in this checkout, see the [inference performance guide](../inference-performance.md)
and [FP16 model preset](../configs/fast-fp16.yaml).

## Versions and availability

| Component | Revision |
|---|---|
| Reference RetroChimera main | `09f2dc75a402d3df82ba51a066aaa852950e9cd4` |
| Canonicalization-batching model code | `aeb46479447c19255c84d9068313b0a4fdd8c94d` |
| Syntheseus, including native concurrent search | `7905e45ef7de5d2fac2597f46dd0ad60412015e9` |
| Experimental model-code port | `68d179387b1c591041937475e6a2b38e553c2bfe` — unpublished benchmark source, **not included in this branch** |

The experimental port added finished-beam CPU bookkeeping, fused/shared-beam attention, and a
custom CUDA KV gather/append implementation to the canonicalization-batching code. These are
**code changes**, not features enabled by the FP16 preset. Their benchmark source is not
published with this record, so those experimental results are not independently reproducible
from this branch. No obsolete RetroChimera-local search broker or target-local RNG patch was used.

## Configuration matrix

All learned weights remained FP32. The precision column is the transformer's wrapper-managed
CUDA autocast selection; Torch's FP32 matmul policy is a separate, process-wide setting.

| Configuration | Transformer `inference_precision` | Torch FP32 matmul policy | Predictions per canonicalization task | Attention / decoder cache | Required model code |
|---|---|---|---:|---|---|
| Reference FP32 | `float32` | `highest` | 1, unchunked | Legacy / legacy | Reference main |
| FP16 + chunking | `float16` | `highest` | 8 | Legacy / legacy | Canonicalization batching; supplied preset |
| Optimized FP16 | `float16` | `high` | 8 | Requested fused, **actual FP16 fallback to legacy** / legacy | Plus experimental CPU bookkeeping |
| Full BF16 | `bfloat16` | `high` | 8 | Fused / CUDA | Plus experimental bookkeeping, attention and cache code |

In the experimental source, backend options were:

- Optimized FP16: `attention_backend="fused"`, `decoder_cache_backend="legacy"`.
- Full BF16: `attention_backend="fused"`, `decoder_cache_backend="cuda"`.

Those backend keywords are **not supported in this branch**. The experimental fused-attention
and cache integration explicitly excluded FP16; no guards were bypassed. Untimed real-model
warmup verified zero fused/custom-cache dispatch for FP16, and 856 fused-attention / 416 custom
cache helper calls for full BF16. Observation wrappers were removed before timed evaluation;
these are warmup counts, not counts across the timed dataset.

Precompiled/AOT artifacts were not counted as another steady-state optimization: normal
model-side prebuilt selection was not available. Loading, compilation and warmup were excluded
from model-call timing, so this is not cold-start or production-image delivery evidence.

## Common model and runtime settings

| Setting | Value |
|---|---|
| Model scope | Full Pistachio `RetroChimeraModel`: transformer **and** template localization |
| Ensemble execution | `call_submodels_in_parallel=True`, CUDA streams |
| Transformer | Beam 20, ten requested augmentations, duplicate-augmentation filtering enabled, maximum generated length 512 |
| Canonicalization | 16 processes; task size from the matrix; reusable Loky workers |
| Template/rule execution | 16 processes |
| Prediction caches | Parent and both submodel `ReactionModel` caches disabled for fixed inference |
| GPU | One RTX A6000 / SM86, the same device for every timed run |
| Runtime | Python 3.12.13, Torch 2.8.0+cu128, RDKit 2025.09.2 |
| Torch CPU threads | 8 |
| Worker environment | `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, `OPENBLAS_NUM_THREADS=1` |
| CPU affinity | Logical CPUs 0–15 and 64–79 on the measured host; tune for your allocation |
| cuDNN TF32 | `torch.backends.cudnn.allow_tf32=False` throughout |
| Other model settings | Defaults of the frozen revisions, including ensemble/rank aggregation policy |

Other GPU jobs were not paused or modified. GPU ownership was monitored; no foreign process
appeared on the measured GPU. Fixed affinity and host monitoring do not establish exclusive
host-CPU use.

## Fixed-inference workload and timing

- Exact matched **1,984-record ordered test prefix**, 124 batches of 16, top-50 results.
- Official evaluation settings: `fold=TEST`, `batch_size=16`, `num_top_results=50`,
  `num_dataset_truncation=1984`, `skip_repeats=True`, `num_gpus=1`,
  `include_predictions=True`, `save_outputs=True`, `resumable=True`,
  `print_idxs=[1,3,5,10,50]`.
- Initialize with Syntheseus `set_random_seed(0)` before model loading. Warm up once with
  `Molecule("Oc1ccc(OCc2ccccc2)c(Br)c1")`, requesting 50 results, and synchronize CUDA.
- Retain native augmentation/translator RNG behavior; **do not reseed each batch**. Input,
  ground-truth, augmentation and RNG hashes matched across every evaluated batch.
- Synchronize CUDA immediately before and after each normal full-ensemble API call. Sum call
  times, excluding loading/warmup, dataset iteration, metric calculation and output writing.
  The standard CLI example selects model settings but is not this timing harness.
- Order: reference FP32, FP16 + chunking, optimized FP16, full BF16, reference FP32 repeat.
  One full pass per optimized configuration; bracket with two FP32 controls.
- Accuracy was recomputed from saved predictions and checked against official statistics.
  Checkpoint, source, harness and input hashes were checked before/after the experiment.

### Measured full-ensemble results

The FP32 time is the mean of the two controls; their timing spread was **0.253%**.

| Configuration | Model-call time | Throughput vs current FP32 | Absolute top-1 accuracy | Absolute top-50 accuracy | Peak PyTorch allocated / reserved |
|---|---:|---:|---:|---:|---:|
| Reference FP32 | 1,302.21 s | 1.000× | 46.9758% | 93.0948% | 9.60 / 26.35 GiB |
| FP16 + chunking | 834.97 s | 1.560× | 46.9758% | 93.0948% | 7.01 / 19.50 GiB |
| Optimized FP16, experimental | 743.02 s | 1.753× | 46.9758% | 93.0948% | 7.02 / 19.50 GiB |
| Full BF16, experimental | 500.02 s | 2.604× | 46.8750% | 92.9940% | 6.95 / 19.28 GiB |

The 1.560× result combines FP16 and chunking; it does **not** isolate the speedup of chunking.
The extra 12.4% optimized-FP16 throughput combines bookkeeping and TF32, not an individual
ablation. Configuration alone on this branch does not deliver 1.753× or 2.604×.

Accuracy is agreement with labeled answers, not agreement with FP32. Relative to the reference:

- Both FP16 cases preserved every top-1/top-50 correctness flag; top-3/top-5 each lost one correct
  record (−0.0504 percentage points), while top-10 gained one (+0.0504 pp).
- Full BF16 lost two correct records at both top-1 and top-50 (−0.1008 pp each).
- Ranked predictions were not identical: even the two FP32 controls differed on seven lower-ranked
  lists despite matching every reported correctness flag. Similar accuracy is not numerical or
  route equivalence.

This runtime differs from the earlier Python 3.9/RDKit 2023 precision-only image; its FP32 top-1
count was 933 rather than this run's 932. That one-record difference was not causally isolated.
Use fresh same-run controls; do not combine absolute accuracy or multiply speedup ratios across
these experiments. A selected prefix and one candidate pass are not general accuracy or
performance certification.

## Native Syntheseus search configuration

Each configuration also ran two opposite-order sweeps over all **200 deployment targets**:

| Setting | Value |
|---|---|
| Model profile | Same ensemble, precision, worker and Torch settings as above; top-20 results |
| Search | Native Syntheseus CLI/broker, Retro*, depth 20 |
| Admission/batching | `max_active_searches=32`, `inference_batch_size=16`, `inference_batch_wait_s=0.01` |
| Per-target budgets | `time_limit_s=30`, `limit_reaction_model_calls=1000`, `limit_iterations=100000` |
| Caching | Physical backend/submodels disabled; `reaction_model_use_cache=True` for per-target facades |
| Inventory | 812,020 entries; prepared once with `SmilesListInventory(canonicalize=True)`, then loaded with `canonicalize_inventory=False` |
| Feasibility | Separate GINE-PNA model, threshold 0.55, added at the model-factory boundary; **not supplied by the preset** |
| Outputs | No route plots; save per-target statistics and graph pickles |
| Other search hyperparameters | Native RetroChimera defaults from [the pinned Syntheseus configuration](https://github.com/microsoft/syntheseus/blob/7905e45ef7de5d2fac2597f46dd0ad60412015e9/syntheseus/cli/search_config.yml) |

The factory hook added feasibility filtering and untimed warmup; native broker scheduling,
search algorithms and output saving were unchanged. Search timing includes per-target graph
and statistics writes, excludes model loading/warmup/inventory preparation and subsequent graph
readback. The model-only example in the guide does not reproduce this external feasibility setup.

| Configuration | Solved / 200, two sweeps | Median total search wall | Uncached molecule-query calls, two sweeps |
|---|---:|---:|---:|
| Reference FP32 | 24, 22 | 321.67 s | 411, 411 |
| FP16 + chunking | 29, 29 | 227.14 s | 444, 452 |
| Optimized FP16, experimental | 47, 46 | 262.90 s | 593, 595 |
| Full BF16, experimental | 64, 68 | 240.95 s | 853, 836 |

These time-limited searches perform different amounts of work. Running inference is not
interrupted at the budget, and queueing affects per-target time. More solves within budget is
not a 2.604× route-search latency claim. Uncached molecule queries are not GPU-batch counts;
solve sets also changed on same-revision reruns. Learned feasibility plus inventory closure does
not independently establish chemical correctness.

All 1,600 target statistics files and graph pickles were read back, with matching root identity,
graph size and solved status. The experimental code passed 110 focused GPU tests. No A100/Hopper,
production-image deployment, equal-call-budget search comparison or general route-quality
certification is established by this refresh.

## Input fingerprints

The benchmark's labeled inputs and deployment feasibility assets are not distributed with this
repository. These fingerprints identify the recorded experiment, not public prerequisites for
using the fast profile. Supply your own approved checkpoint, data, inventory and filters.

| Asset | SHA-256 |
|---|---|
| Pistachio checkpoint archive | `b0d30506d780693b293991cf2e5c5ac0be773748ccddbcd46684de5696d8a60f` |
| 1,984-record prefix | `453ad4da8bcec2aec68c82cb91a7345ca44ce1688f9f16fa84576b3eb6cf87d9` |
| 200 search targets | `f32caa8d426a7a46e0213aa1e445f17017cb5789b43f1e3f3e3809008e9e816b` |
| Canonicalized inventory | `e84dfd9bcb76770d963170e5006703e16c2765a2258a6a74c29dcc243eecd62e` |
| Feasibility artifact | `fab64c649aa2b3733cff7fc7b5e8348df0dae2842567b9217b8731411cdfaff6` |

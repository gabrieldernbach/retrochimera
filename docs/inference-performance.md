# Inference performance

This guide provides an **opt-in FP16 profile** for the full Pistachio ensemble. It does not
change the default FP32 policy, training, or search defaults. Reduced precision can change
rankings and search outcomes; validate accuracy on your workload before adopting it.

## FP16 model profile

The reusable [fast-fp16.yaml](configs/fast-fp16.yaml) selects:

- Parallel transformer/template inference on CUDA.
- Transformer `inference_precision: float16`, beam size 20, ten requested augmentations,
  duplicate-augmentation filtering, 16 canonicalization workers, and chunks of eight predictions.
- 16 template/rule workers, with submodel prediction caches disabled for uncached measurements.

The nested model names, `smiles_transformer` and `template_localization`, must match the
checkpoint's `models.json`. Adapt them for other ensemble layouts; this is not a standalone
Transformer or forward-model preset.

Run the examples from the repository root, using the code in this checkout rather than an older
installed release that may not support `canonicalization_chunksize`. Use the
[installation instructions](../README.md#using-retrochimera) to prepare your environment, then
install this checkout with `pip install -e .` in your CUDA-enabled environment.

**User inputs:** select an available GPU via `CUDA_VISIBLE_DEVICES`; set `MODEL_DIR` to the
full ensemble checkpoint directory. For the relevant examples, also set `TARGETS_FILE` (one
product SMILES per line), `DATA_DIR` (preprocessed evaluation dataset), `INVENTORY_FILE`, and a
fresh `RESULTS_DIR`. `cuda:0` below means the first visible GPU, not necessarily physical GPU 0.

### Python inference

This example uses the measured FP16 runtime policy: eight Torch threads, FP32 matmul precision
`highest` (not TF32 `high`), and cuDNN TF32 disabled. Model weights remain FP32; the transformer
uses CUDA autocast rather than permanently converting the weights to FP16.

```python
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from retrochimera import RetroChimeraModel
from syntheseus import Molecule
from syntheseus.reaction_prediction.utils.misc import set_random_seed

set_random_seed(0)
torch.set_num_threads(8)
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False

profile = OmegaConf.load("docs/configs/fast-fp16.yaml")
model = RetroChimeraModel(
    model_dir=os.environ["MODEL_DIR"],
    device="cuda:0",
    use_cache=False,
    **profile.model_kwargs,
)

# Untimed warmup; keep native augmentation/translator RNG behavior between batches.
model([Molecule("Oc1ccc(OCc2ccccc2)c(Br)c1")], num_results=50)
torch.cuda.synchronize()

products = [
    Molecule(line.strip())
    for line in Path(os.environ["TARGETS_FILE"]).read_text().splitlines()
    if line.strip()
]
for start in range(0, len(products), 16):
    predictions = model(products[start : start + 16], num_results=50)
    print(f"Predictions for {len(predictions)} products")
```

Batch size 16 and top-50 are benchmark settings, not requirements or production recommendations.
See the [model-use limitations](../README.md#checkpoints-for-retrochimera-1) before using predictions.

### Evaluation CLI

The same model profile can be loaded with `--config`. These CLI examples were checked against
Syntheseus `7905e45`; older versions may lack resumable evaluation or concurrent-search options.
The profile itself does not require the concurrent-search API.

```sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1

python -m retrochimera.cli.eval \
    --config docs/configs/fast-fp16.yaml \
    model_dir="$MODEL_DIR" \
    data_dir="$DATA_DIR" \
    results_dir="$RESULTS_DIR" \
    model_kwargs.use_cache=False \
    num_gpus=1 fold=TEST batch_size=16 num_top_results=50 \
    float32_matmul_precision=highest \
    resumable=True
```

On completion, the resumable evaluator writes `resumable_TEST/stats.json` and
`resumable_TEST/predictions.jsonl` under `RESULTS_DIR`. Use a new results directory for a fresh
measurement: completed resumable outputs are reused, not recomputed.

The CLI selects the model settings but does not reproduce the benchmark's untimed warmup,
explicit CUDA synchronization, or cuDNN policy initialization. Use the Python initialization
above when controlling those settings. Do not treat CLI wall time as the recorded warmed
model-call timing.

### Concurrent search

Search concurrency and cross-target inference batching belong to
[Syntheseus's search CLI](https://github.com/microsoft/syntheseus/blob/main/docs/cli/search.md#concurrent-targets-and-batched-inference).
Use a Syntheseus version containing those options; no RetroChimera-local broker is needed.

```sh
python -m retrochimera.cli.run_search \
    --config docs/configs/fast-fp16.yaml \
    model_dir="$MODEL_DIR" \
    search_targets_file="$TARGETS_FILE" \
    inventory_smiles_file="$INVENTORY_FILE" \
    results_dir="$RESULTS_DIR" \
    use_gpu=True num_top_results=10 \
    max_active_searches=32 inference_batch_size=16 inference_batch_wait_s=0.01 \
    time_limit_s=30 limit_reaction_model_calls=1000 limit_iterations=100000 \
    retro_star_config.max_expansion_depth=20 \
    reaction_model_use_cache=True num_routes_to_plot=0 save_graph=True
```

This is a **model-only** search example. It does not install a feasibility filter; the recorded
benchmark instead used top-20 with a separate feasibility model at threshold 0.55. Follow the
model-use limitations and select appropriate filtering for your application.

The preset intentionally omits the parent `model_kwargs.use_cache`: concurrent search configures
the physical backend cache itself and keeps per-target caches via `reaction_model_use_cache`.
Do not add the evaluation-only parent-cache override to the search command. Likewise,
`float32_matmul_precision` is an evaluation option, not a search CLI field; set Torch's policy
before invoking search from Python if embedding it in an existing process.

## Tuning and fallbacks

- FP32 remains the default. To return to the reference arithmetic policy, select transformer
  `inference_precision="float32"` and Torch matmul precision `"highest"`, without an enclosing
  caller-managed autocast context. The wrapper's FP32 option does not cancel such a context.
- `torch.set_float32_matmul_precision("high")` allows TF32 for eligible FP32 matmuls. It is separate
  from transformer FP16 autocast. In the evaluation CLI, use `float32_matmul_precision=high`.
  This can change numerical results and is not the measured FP16 preset's `highest` policy.
- The backward and forward Transformer wrappers share `canonicalization_chunksize`. It must be
  a positive integer, not a Boolean; the default is eight. Set it to one for unchunked scheduling.
  Chunking preserves canonicalized outputs, order, and scores, including invalid-SMILES handling.
  It uses the existing reusable Loky pool; idle workers retire after five minutes.
- Worker counts and batch size are workload/hardware settings, not universal optima. Reduce them
  for smaller CPU/GPU allocations. In the recorded FP16 run, peak PyTorch allocation was about
  7.01 GiB but peak allocator reservation was 19.50 GiB; the allocation figure is not a GPU-size
  requirement or a memory-fit guarantee.
- Cache disabling measures uncached inference; it is not necessarily fastest for repeated inputs.
  Choose the cache policy appropriate for your serving workload.

## Benchmark scope and experimental variants

See the [benchmark configuration record](benchmarks/inference-2026-10-28.md) for all precision,
matmul, model, runtime, and search settings used in the comparison.

The supplied preset matches the **FP16 + canonicalization batching** case available in this
checkout. The faster experimental cases also contained unmerged code: finished-beam bookkeeping,
and, for BF16, fused/shared-beam attention plus a custom CUDA KV cache. Configuration alone on
this branch does not enable those implementations or reproduce their reported speedups.

In that experiment, the fused-attention and CUDA-cache integration explicitly excluded FP16.
Their `attention_backend` / `decoder_cache_backend` options are **not accepted by this branch**;
do not add them to this preset. FP16 must not be described as running those kernels when it is
falling back to legacy attention/cache behavior.

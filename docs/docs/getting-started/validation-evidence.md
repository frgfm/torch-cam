# Inspect validation examples with an agent

Accuracy can improve while individual predictions still deserve investigation. The
[`validation_cam.py` example](https://github.com/frgfm/torch-cam/blob/main/scripts/validation_cam.py)
selects a small set after normal validation and uses existing batched `GradCAM` extraction.
It adds no library API, training callback, or dependency.

## Run the example

Install the repository with `uv pip install -e '.[scripts]'`. For the pretrained torchvision
CLI, put images in folders named with their **model output indices**, not a new contiguous
label ordering. For example, ResNet18's `207/` contains golden retrievers and `281/` contains
tabbies. Subsets work because folder names are converted to those actual indices.

```text
validation-images/
  207/
    dog.jpg
  281/
    cat.jpg
```

```bash
python scripts/validation_cam.py validation-images evidence/epoch-001 \
  --arch resnet18 --target layer4 --batch-size 16 --per-bucket 4 --seed 0
```

Use `--device cuda` or `--device mps` when available. Use a new output directory each epoch;
existing directories are never overwritten. The CLI downloads the model's default weights
if absent. Model loading and downloads are outside the recorded validation timings.

The selection retains up to `--per-bucket` examples from each bucket:

| Bucket | Selection | Class maps |
| --- | --- | --- |
| `wrong_high` | Highest softmax confidence among wrong predictions | Predicted class and labeled class |
| `correct_low` | Lowest softmax confidence among correct predictions | Predicted/labeled class and runner-up |
| `control` | One seeded random example per class, up to the bucket limit of distinct classes, independent of predictions | Same comparison rule as above, according to correctness |

Samples in multiple buckets are explained once, with every selection reason recorded.
The control panel stays fixed across epochs when dataset order, labels and seed repeat,
including when the model's predictions change. Confidence is a selection heuristic, not calibrated
certainty. Empty buckets are normal. Selection holds only the bounded high/low lists and
one control record per class, not all images or logits.

## Inspect the evidence

`contact-sheet.png` has one row per case: model-view input, predicted-class overlay,
and labeled-class or runner-up overlay. Captions are shortened to fit; metadata has full names.

`report.json` is the completion marker. It records selection reasons, dataset indices,
relative sample IDs, true/predicted/runner-up classes and probabilities, compared classes,
blank-map flags, artifact locations, settings, versions, and timings. Each `sample-*`
directory reuses the existing [prediction explanation bundle](debug-prediction.md#result-contract):
model-view pixels, raw float32 NumPy maps, heatmaps, overlays, logits and class metadata.
The labeled class remains `expected` even when the second map explains the runner-up.

Ask the agent to inspect confident errors, uncertain successes, and controls together.
A map highlighting a timestamp or background suggests a hypothesis to test; it does not
prove a semantic bias or causal influence. Blank maps are uninformative, not evidence that
a feature is absent. Maps are independently normalized: do not compare color intensity
as an absolute contribution across classes or epochs. Confirm labels before proposing
training changes. Add analogous counterexamples from a separate training/acquisition pool;
keep validation examples out of training and reserve a final untouched test set.

## Use your own training loop

The two helpers live in the **example script**, not the supported package API or installed
wheel. The snippet assumes the TorchCAM checkout is on your Python import path. Otherwise,
copy and adapt the selection and batched extraction code into your validation routine.
For an `ImageFolder` using your trusted, deterministic preprocessing and class mapping:

```python
from torch.utils.data import DataLoader
from scripts.validation_cam import validate, save_evidence

# model, dataset, class_names and model_view come from your training repository.
# model_view(image) applies inference's spatial transforms before normalization.
model.eval()
loader = DataLoader(dataset, batch_size=16, shuffle=False)
cases, validation = validate(model, loader, per_bucket=4, seed=0)
save_evidence(
    model, dataset, cases, "evidence/epoch-001",
    class_names=class_names, model_view=model_view,
    target_layer="backbone.layer4", batch_size=16,
    context={"checkpoint_id": "epoch-001"}, validation=validation,
)
# Your training loop decides when to call model.train() again.
```

Normal validation runs without gradients. Selected inputs are reloaded and forwarded once
per CAM batch, with two class-target gradient passes through that forward. Parameter
gradients and training/evaluation flags are preserved, and hooks are removed on failure.
CAM extraction can run under `torch.no_grad()` but must be outside `torch.inference_mode()`.
The model must have independent batch samples and finite `(batch, classes)` logits with at
least two classes. This example uses one target layer. Inference must stay unchanged
between selection and extraction; changed predictions/probabilities cause an error rather
than saving mismatched evidence. Use the exact model-view crop for the saved input.

## Measure the added work

The console and `report.json` separate:

- `validation_seconds`: validation, data loading and selection together;
- `selection_seconds`: the confidence calculation, CPU transfer and online shortlist work
  inside validation (a subset of `validation_seconds`, not a separate baseline run);
- `cam_seconds`: target-layer setup, selected-input loading, forwards, two class-target
  gradient passes and CPU map transfer;
- `artifact_seconds`: model-view loading, per-example artifacts and contact-sheet rendering.

CUDA/MPS work is synchronized at timing boundaries. Final report serialization is excluded.
The example reports observed wall time, not a speedup. For a matched overhead comparison,
measure your normal validation loop and validation plus these diagnostics on the same
checkpoint, data, batch size, device and threads. Warm up both paths, alternate their order,
repeat, and report all trials. Keep generated images and raw timing reports outside Git.

### Measured example overhead

On an Apple M3 Pro, macOS 15.8.1, Python 3.11.13, torch 2.15.0 and torchvision 0.29.2:
ResNet18 `IMAGENET1K_V1`, 224px crops, batch 16, one CPU/intra-op thread and one inter-op
thread, `layer4`, seed 0, two cases per bucket (six distinct selected cases). The 96-input
timing fixture repeats each of the two torchvision gallery dog photos 12 times in each
of four forms: original, mirrored, Gaussian blur (radius 8), and downsampled to 16px then
resized back with Pillow defaults. It is an integration/timing fixture, not an accuracy benchmark.
Five warmed paired trials alternate baseline/diagnostic order; GPU work is synchronized.

| Device | Normal validation | Validation + selection + all evidence | Selected CAM work | Artifact work |
| --- | ---: | ---: | ---: | ---: |
| CPU | 4.706 s | 5.029 s | 0.139 s | 0.149 s |
| Apple GPU (MPS) | 0.537 s | 0.732 s | 0.041 s | 0.148 s |

Columns are independent medians. Paired additional wall time ranges were 0.142–0.450 s
(CPU) and 0.190–0.198 s (MPS). These include disk output and are not universal overhead
estimates. Rendering dominated CAM work on MPS; reduce the selected set or save evidence
less often when integrating it into a training loop.

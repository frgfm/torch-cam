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

## Compare a horizontal flip

Add `--compare-flip` to compare the selected **initially correct** cases with horizontally
flipped model inputs. Only use it when flipping preserves the label. No augmentation
registry or automatic bias verdict is introduced.

`flip-contact-sheet.png` shows the original and flipped inputs, signed original CAM,
signed flipped CAM aligned back to original coordinates, and their difference. Rows with
a changed top-1 class come first; stable-top-1 rows follow in descending CAM distance.
Unavailable distances come last within each group.
Full probability drift is shown alongside CAM drift, so stable top-1 is not mistaken for
stable probabilities. The panel covers the bounded selection, not every validation case.

For each pair, the objective is `label logit - strongest incorrect flipped-class logit`.
The same comparison class is used for both views. Two opposite `targets` calls with
`normalized=False` recover signed attribution before the base GradCAM ReLU: positive
direction minus negative direction. The flipped map is unflipped before comparison.
Red/blue means positive/negative attribution to that score difference, not semantic
correctness. Signed overlays use a grayscale background and a shared symmetric scale
within each pair; the difference image uses that same scale. Scales differ between pairs.

Each sample's `flip` metadata records the class pair, margins, flipped prediction,
probabilities, and two separate measurements:

- `probability_tv`: half the sum of absolute changes across **all** class probabilities;
- `aligned_cam_distance`: half the L1 distance between the two signed maps after dividing
  each by its own absolute L1 mass. It ranges from 0 (identical normalized maps) to 1.
  If either map has zero L1 mass, the distance is `null`; raw masses remain available in
  `cam_l1_norms`. Small masses and method instability can make drift uninformative.

The distance measures normalized spatial patterns; the change image uses raw signed
values. A pure scale change can therefore alter that image while leaving distance zero.

`flip-data.npz` stores float32 logits (original, flipped) and both raw signed maps in
original coordinates (`original`, `aligned`). Load it with `np.load(..., allow_pickle=False)`.
Per-pair images and scores permit inspection without treating CAM drift as a model defect.

### Controlled example: identical probabilities, different reliance

```bash
python scripts/flip_reliance_example.py evidence/controlled-flip
```

The declared label is "red object present"; a green patch is irrelevant. A hand-built
neural control reads whichever red/green cue occupies a left-hand position. Both original
and flipped inputs return exactly the same logits and probabilities (95.26% for the label),
but aligned CAMs switch from the red object to the green cue: probability TV is 0 and CAM
distance is 1. Removing green leaves the original prediction unchanged, but flips the
augmented prediction to "red absent". `controlled-check.json` records all four conditions.
The script checks the probability identity and nuisance-removal class change. Tests also
check the map displacement and distance. The existing scripts workflow runs it and uploads
`flip-reliance-results`, including the panel and the numerical checks. This is a controlled diagnostic, not a trained-model
or generalization benchmark; its explicitly known label rule establishes the nuisance cue.

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
    context={"checkpoint_id": "epoch-001"}, validation=validation, compare_flip=True,
)
# Your training loop decides when to call model.train() again.
```

Normal validation runs without gradients. Selected inputs are reloaded and forwarded once
per CAM batch, with two class-target gradient passes through that forward. Parameter
gradients and training/evaluation flags are preserved, and hooks are removed on failure.
Flip comparison adds one forward on both original and flipped initially correct samples
(at most twice the CAM batch size), followed by two target-gradient passes. It reuses the
loaded inputs. Extra signed overlays, pair data and the paired sheet add rendering/I/O work.
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
- `cam_seconds`: target-layer setup, selected-input loading, forwards, class-target
  gradient passes and CPU map transfer, including optional flip comparison;
- `artifact_seconds`: model-view loading, per-example artifacts and contact-sheet rendering.

CUDA/MPS work is synchronized at timing boundaries. Final report serialization is excluded.
The example reports observed wall time, not a speedup. For a matched overhead comparison,
measure your normal validation loop and validation plus these diagnostics on the same
checkpoint, data, batch size, device and threads. Warm up both paths, alternate their order,
repeat, and report all trials. Keep generated images and raw timing reports outside Git.

### Measured example overhead

Measured on 2026-11-01 with an Apple M3 Pro, macOS 15.8.1, Python 3.11.13, torch 2.15.0 and torchvision 0.29.2:
ResNet18 `IMAGENET1K_V1`, 224px crops, batch 16, one CPU/intra-op thread and one inter-op
thread, `layer4`, seed 0, two cases per bucket (six distinct selected cases). The 96-input
timing fixture repeats each of the two torchvision gallery dog photos 12 times in each
of four forms: original, mirrored, Gaussian blur (radius 8), and downsampled to 16px then
resized back with Pillow defaults. It is an integration/timing fixture, not an accuracy benchmark.
Five warmed trials rotate baseline/raw-evidence/flip-evidence order; GPU work is synchronized.
Six cases are selected, with three initially correct cases receiving flip comparisons.

| Device | Normal validation | With raw evidence | With raw + flip evidence |
| --- | ---: | ---: | ---: |
| CPU | 5.186 s | 5.363 s | 5.772 s |
| Apple GPU (MPS) | 0.561 s | 0.768 s | 0.893 s |

Columns are independent medians and include all disk output. The median paired flip-only
increment was 0.366 s on CPU (range 0.219–0.609 s) and 0.114 s on MPS
(0.103–0.143 s). CAM work changed from 0.148 to 0.262 s on CPU and from 0.042 to
0.056 s on MPS. Artifact work changed from 0.167 to 0.273 s on CPU and from 0.153 to
0.260 s on MPS. Rendering dominates the extra MPS cost. These are fixture-specific
measurements; select fewer cases or save less often when integrating into a training loop.

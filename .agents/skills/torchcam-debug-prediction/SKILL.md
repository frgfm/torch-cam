---
name: torchcam-debug-prediction
description: Debug surprising 2D image-classification predictions and investigate suspected shortcuts in an existing PyTorch repository with TorchCAM. Use for predicted-versus-expected CAM bundles, failures and successful controls, controlled cue checks, or bounded training/data experiments with independent repair verification. Reuse the owner's trusted checkpoint, preprocessing, labels, and training pipeline. Separate cue dependence, unresolved location evidence, and verified repair outcomes.
compatibility: Requires local Python execution, PyTorch, Pillow, and torchcam>=0.5.0.
metadata:
  author: frgfm
  version: "1.1"
---

# Debug predictions and investigate shortcuts with TorchCAM

Produce reproducible visual evidence for one surprising classifier prediction. CAMs show class-associated activation, not why the model decided, causal influence, correctness, or localization quality.

For a suspected shortcut or repair request, also follow steps 6–9 and read [the investigation contract](references/shortcut-investigation.md) before recommending an intervention. The prerequisite experiment found **no CAM-guided repair advantage**; use this workflow for investigation and bounded experiments, with no promise of repair.

## 1. Discover the repository's inference path

Find and reuse the code that already defines:

- the model architecture and trusted checkpoint loader;
- evaluation preprocessing, including resize, crop, normalization, and color conversion;
- the ordered class-name mapping;
- the original image before normalization.

Record the checkpoint hash, loader and preprocessing entry points, class ordering, runtime, and baseline logits on fixed examples. Check the diagnostic tensor against the owner's evaluation tensor before interpreting CAMs. Resolve a proven diagnostic preprocessing mismatch in the diagnostic helper, then rerun; preserve the trusted inference path. Keep the training entry point and configuration for a possible later experiment.

Search the repository before writing code. Prefer its test or inference entry point over reconstructing the model. Load only checkpoints already trusted by the owner; TorchCAM does not load checkpoints or preprocessing.

Confirm the model returns one logits tensor shaped `(1, num_classes)`. If it returns a tuple or dictionary, wrap it in a small `nn.Module` that selects the logits tensor without changing inference.

## 2. Choose the extractor

- CNN: start with the default `GradCAM`. Let TorchCAM resolve the last spatial layer. If resolution fails or the repository has a known semantic feature layer, pass that exact module or name as `target_layer`.
- Torchvision Vision Transformer: use `LeGrad` and explicit transformer blocks, usually the last four: `list(model.encoder.layers)[-4:]`.
- Other ViTs: use `LeGrad` only when the blocks expose supported batch-first `nn.MultiheadAttention`. Pass the repository-specific `score_projection`, `prefix_tokens`, or `grid_shape` through `method_kwargs` when required.
- Existing reshape-based CAM setup: preserve its explicit target layer and pass `reshape_transform` through `method_kwargs`.

Do not add an architecture registry or guess a ViT configuration.

## 3. Run one explanation

Call `model.eval()`. Keep the call outside `torch.inference_mode()` because CAM extraction needs gradients. The input must be a batch of one shaped `(1, C, H, W)`.

```python
from torchcam.explain import explain

result = explain(
    model,
    input_tensor,
    expected_class_idx=expected_idx,
    class_names=class_names,
    target_layer=target_layer,  # omit for automatic CNN resolution
)
bundle = result.save("torchcam-explanation", image=original_image, alpha=0.5)
```

For a ViT, also pass `method=LeGrad` and the explicit blocks. Use a new output directory; `save` refuses to overwrite an existing one.

## 4. Validate the evidence bundle

Treat `manifest.json` as the completion marker. Before reporting success:

1. Parse it and require `schema_version == 1`.
2. Require a prediction reference and nonempty `manifest["classes"]`, a dictionary keyed by string class index. Each class entry's `artifacts` is a nonempty **list** of per-layer dictionaries: iterate that list before reading `artifact["map"]`, `artifact["heatmap"]`, or `artifact["overlay"]`. Resolve every relative path under the bundle directory and require each file to exist. Prediction and expected entries contain class references; logits and probabilities live in the corresponding class entry.
3. Load each `.npy` map with `allow_pickle=False`; require a finite two-dimensional `float32` array.
4. Open every overlay and require its dimensions to match `manifest.json`'s `image_size`.
5. Confirm the prediction and optional expected references match the corresponding class entries and the repository's ordered labels.

No manifest means the bundle is incomplete, even if some images exist.

## 5. Report to the owner

Return:

- predicted class and the owner's expected class, with indices and model scores/probabilities from the manifest;
- the resolved method and target layers;
- the bundle path and the most useful overlay paths;
- any compatibility boundary or uncertainty encountered.

Use language such as “the GradCAM map highlights…” or “activation differs between…”. Do not claim the map proves causation, model correctness, object localization, bias, safety, or compliance.

Check each map's range before describing it. If it is constant or all zero, call it a blank map and do not claim it highlights a region or shows positive class-associated activation; a blank CAM is not proof that a feature is absent.

If extraction fails, use the TorchCAM prediction-debugging guide and troubleshooting page before changing repository model code.

## 6. Inspect failures with successful controls

Freeze a discovery set containing failures and successes, with labels, sample IDs, and relevant group counts. Use comparable successful controls (same class/task, different suspected cue where available). Explain each image separately; preserve blank maps and extraction errors in the denominator. Check alignment with the model's actual crop before describing a region.

## 7. Test a cue hypothesis

Write a falsifiable hypothesis naming the suspected cue, expected score/error change, task-preserving edit, matched controls, and disconfirming result. A highlighted corner is a proposal. Confirm on separate validation examples with controlled edits, logging original and edited logits/probabilities, accuracy, groups, edit magnitude, and label preservation. Use training-only donors; keep test data out of selection.

Distinguish observed cue sensitivity, supported location-specific findings, and unresolved evidence. Matched controls can change globally pooled cue evidence too; a failed location gate does not exclude cue dependence. Conversely, a blank map or a null check does not prove that no shortcut exists. Stop at an unresolved finding when edits cannot preserve labels or controls are inadequate.

## 8. Choose a bounded experiment

Check the owner's existing authorization for local training and data edits. When evidence supports an experiment, reuse the owner's trainer, labels, preprocessing, and approved data in a separate output directory. Freeze the change, seeds, step/example budget, validation selection rule, success criteria, and regression tolerances before scoring. Compare unchanged continuation and ordinary augmentation at the same training budget; include a no-shortcut control when available.

Prefer the smallest experiment that tests the hypothesis. Mark a CAM-selected edit exploratory if confirmation is unresolved; do not recommend it as a supported repair. The reference notebook's provisional fallback is an experimental arm, not a default remediation policy. Do not broaden data collection, relabel data, replace the deployed checkpoint, or deploy a repair beyond the owner's authorized scope.

## 9. Verify and report the outcome

Freeze candidate selection before using independent, untouched evaluation data. Score the original checkpoint and every experiment arm on average, per-class, and relevant group/worst-group accuracy with counts; retain seed-level results, comparator differences, regressions, and failures. If evaluation informed further tuning, use fresh evaluation data and disclose that change.

Verify the original inference tensor, logits, checkpoint, and code remain unchanged by investigation; candidate weights belong in separate artifacts. A prettier CAM or validation gain is not repair verification. Report a repair only against the frozen success criteria and comparators; otherwise report failed, unresolved, or improved without evidence of an intervention advantage.

Save the investigation record described in the reference alongside schema-v1 explanation bundles. Return the hypothesis, evidence for/against it, authorization and budget, independent evaluation results, regressions, unsupported claims withheld, and artifact paths. Keep deployment a separate owner decision.

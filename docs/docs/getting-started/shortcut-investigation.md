# Investigate a suspected shortcut with an AI agent

Use the [`torchcam-debug-prediction` skill](https://github.com/frgfm/torch-cam/tree/main/.agents/skills/torchcam-debug-prediction)
to extend a [prediction explanation](debug-prediction.md) into an investigation across failures and successful
controls. Reuse the model owner's checkpoint, evaluation preprocessing, class ordering, and training pipeline.
TorchCAM supplies activation evidence; controlled checks and independent evaluation supply the additional evidence
needed to consider a training or data experiment.

## What the prerequisite actually demonstrated

The completed [shortcut experiment PR](https://github.com/frgfm/notebooks/pull/21) contains an
[executed notebook](https://github.com/frgfm/notebooks/blob/cb719ae40cd137e93cb9a16bdb59f1c03e4ee587/torch-cam/shortcut_repair.ipynb)
with three seeds, a synthetic color cue, separate discovery/confirmation/test sets, matched edits, and a
no-shortcut control. It reuses native PyTorch training and TorchCAM's merged `explain()` API.

| Final arm | Shortcut average / worst-group accuracy | No-shortcut average / worst-group accuracy |
| --- | --- | --- |
| Unchanged continuation | 100% / 100% | 100% / 100% |
| Ordinary augmentation | 100% / 100% | 100% / 100% |
| Provisional CAM-guided erasing | 100% / 100% | 100% / 100% |

Each arm observes 576 optimizer steps and 36,864 presentations. Every confirmation location gate remains
unresolved. All guided-minus-comparator differences and across-seed sample SDs are zero. The initial failed
attempts remain visible, and final test seeds were refreshed after initial tests were viewed.

This demonstrates reproducible investigation and comparison mechanics, **not repeatable CAM-guided repair
value**. The workflow therefore supports bounded hypothesis testing and honest outcome reporting. It does
not prescribe erasing the top CAM region as a validated fix. Accuracy at a ceiling does not prove cue-invariant
logits, and the no-shortcut control rules out only the injected training correlation.

## Ask a coding agent

Run the agent in the repository that owns the model. Identify its existing artifact and authorization scope:

> Investigate these surprising predictions with TorchCAM. Reuse our trusted checkpoint, preprocessing,
> labels, and trainer at `<paths>`. Check diagnostic preprocessing against evaluation first. Compare the
> failures in `<discovery set>` with successful controls, form a cue hypothesis, and test it on separate
> confirmation data using label-preserving edits and matched controls. Distinguish observed cue sensitivity,
> supported location evidence, and unresolved findings. Our authorization covers `<local experiment/data scope>`.
> If evidence supports an experiment, freeze its change, budget, comparators, and regression criteria, reuse our
> trainer, and evaluate the frozen candidates on `<independent data>`. Preserve the original inference path and
> report unsuccessful repairs, comparator ties, and regressions with artifact paths.

The skill's [investigation contract](https://github.com/frgfm/torch-cam/blob/main/.agents/skills/torchcam-debug-prediction/references/shortcut-investigation.md)
defines the decisions and record contents. Check existing authorization before requesting anything additional.
Permission for diagnosis does not by itself permit changing production data or replacing deployed weights.

## Evidence to retain

Check that the diagnostic input equals the owner's evaluation input before interpreting CAMs. Save per-image
[schema-v1 explanation bundles](debug-prediction.md#result-contract) for discovery failures and controls. Preserve
blank maps, extraction errors, sample/group counts, and image-crop alignment. A failed extraction is not a
negative shortcut result.

Freeze cue hypotheses and criteria before confirmation. An edited score change can support cue sensitivity;
location-specific support requires adequate controls. In the notebook, controls can alter globally pooled color
evidence too, so unresolved confirmation cannot exclude a shortcut. Invalid edits that change the task label
cannot justify training intervention.

For an authorized experiment, keep candidate artifacts separate from the original checkpoint. Compare unchanged
continuation and ordinary augmentation under matched training budgets. Freeze selection before independent
evaluation, report average/class/group accuracy and counts, and retain every seed, failed arm, and regression.
Map appearance alone cannot verify a repair.

The notebook exports a separate, unversioned `experiment.json` with protocol/runtime, all diagnoses and
interventions, observed budgets, final weight/test hashes, predictions, group metrics, summaries, and fault probes.
It creates a fresh temporary directory and does not export weights or explanation bundles. Reuse that record and
the notebook's trainer rather than maintaining a second training implementation. Additional checkpoint and
investigation artifacts need their own provenance; the explanation manifest remains unchanged.

## Agent evaluation

The small [paired evaluation](https://github.com/frgfm/torch-cam/tree/main/evals/shortcut-investigation) replays the
same notebook and supplies a double-centering preprocessing error, a genuinely cue-dependent pilot, and a trained
no-shortcut control. It measures correct diagnosis, unsupported intervention recommendations, preservation of the
trusted inference path, and independently checked outcomes, with identical budgets for the original and extended
skill. Its recorded results and limits concern these three synthetic tasks, not general agent or repair performance.

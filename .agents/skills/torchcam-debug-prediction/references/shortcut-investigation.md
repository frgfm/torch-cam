# Shortcut investigation contract

## Prerequisite and limits

Read [notebooks PR #21](https://github.com/frgfm/notebooks/pull/21) and its
[executed notebook](https://github.com/frgfm/notebooks/blob/cb719ae40cd137e93cb9a16bdb59f1c03e4ee587/torch-cam/shortcut_repair.ipynb).
Reuse its functions and exported `experiment.json`; do not copy its trainer.
Its central 16×16 pixels define bar orientation; only the supplied border is editable.
Do not assume that authorization or label-preservation contract applies to owner photographs.

Seeds 7/17/27 compare unchanged continuation, ordinary erasing and provisional CAM-guided erasing:
shared four-epoch pilot, 32 continuation epochs, 576 updates and 36,864 presentations per arm.
Train/discovery/confirmation/test sizes are 1,024/64/256/1,024; test groups have 256 examples each.
CAM and confirmation compute are extra; the no-shortcut regime removes the injected training correlation.

All 18 final outcomes reach 100% average/worst-group accuracy, zero seed SD, and zero guided advantage.
All six location gates remain unresolved. The notebook tests a **provisional** CAM fallback despite that
uncertainty. It validates protocol integrity, not a supported repair policy, CAM's incremental value,
cue-invariant logits, or general bias detection. Historical failed attempts remain visible; final test
seeds were refreshed after initial tests were viewed, and the study was not preregistered.

## Decisions

Freeze proposals on discovery; confirm on separate validation examples; keep independent tests out of
selection. In this notebook both neutralization and training-donor blending must show excess predicted-class
probability drop >0.02 over matched controls and bootstrap lower bound >0. Area and per-image L1/L2 changes
match; label pixels remain untouched. Define and justify another task's criteria before confirmation.

- Proven preprocessing mismatch: fix the diagnostic helper and rerun controls before cue interpretation.
- CAM salience alone, invalid edits or inadequate controls: retain a hypothesis and stop unresolved.
- Cue dependence with unresolved location: report both; any authorized targeted experiment stays exploratory.
- Supported task-preserving checks: consider a bounded experiment through the owner's existing trainer.
- Independent gain with comparator ties or regressions: retain every outcome; do not claim CAM advantage or a
  verified repair that violates frozen regression criteria.
- No supported dependence: retain negative evidence and limits; do not prescribe shortcut-specific training.

Controls can introduce globally pooled cue evidence; an unresolved location gate cannot exclude dependence.
Do not promote a region because one operator passes or its CAM matches a known benchmark cue. Preserve blank
maps and extraction errors. Continue within existing authorization; request only necessary additional scope
for owner data or deployment actions, with a concrete proposal.

## Records

Keep schema-v1 explanation manifests unchanged and validate them using skill step 4. They contain activation
artifacts, not training provenance. The notebook's separate **unversioned** `experiment.json` retains protocol,
runtime, all diagnoses/probes/interventions, observed budgets, final model/test hashes, predictions, group
counts/scores, summaries and fault probes. It exports to a fresh temporary directory; checkpoints and explanation
bundles must be captured separately. Acceptance checks require integrity, not improvement.

Use the owner's record format if one exists. Otherwise retain loader/preprocessor/trainer paths and config,
checkpoint/class-order/input hashes, original logits, runtime and authorization; split IDs/hashes and labels;
frozen hypothesis, edits/controls/donors, magnitude and label checks, decision criteria; per-sample scores,
blank/error statuses and supported/unresolved findings; candidate hashes, seeds, observed budgets and selection;
independent predictions and average/class/group metrics with counts, comparator differences, regressions,
failed repairs and inference-preservation checks. This is not a new TorchCAM API schema.

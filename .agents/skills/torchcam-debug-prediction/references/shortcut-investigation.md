# Shortcut investigation contract

## Read the prerequisite before adapting it

Use the completed [notebooks PR #21](https://github.com/frgfm/notebooks/pull/21) and
[executed notebook at cb719ae](https://github.com/frgfm/notebooks/blob/cb719ae40cd137e93cb9a16bdb59f1c03e4ee587/torch-cam/shortcut_repair.ipynb).
Reuse its functions and exported `experiment.json`; do not copy its trainer into another implementation.
Its synthetic label is bar orientation in the central 16×16 pixels; only its supplied border is editable.
Do not assume this permission or label-preservation contract applies to an owner's photographs.

The three seeds (7/17/27) share a four-epoch pilot, then 32 epochs per arm: unchanged continuation,
ordinary border erasing, and provisional CAM-guided erasing. Each arm observes 576 optimizer steps
and 36,864 training presentations. Training has 1,024 examples, discovery 64, confirmation 256,
and independent test 1,024 (256 per label × cue group). The no-shortcut regime removes the injected
training correlation. CAM and confirmation compute are extra.

All 18 final outcomes reach 100% average/worst-group accuracy, with zero sample SD and zero
guided-minus-comparator differences. All six location diagnoses remain unresolved, even though the
shortcut pilots rank the injected corner first. The notebook intentionally tests a **provisional**
fallback despite unresolved confirmation. It does not validate a supported location repair policy.
The initial attempt's 18 failed outcomes and 24 interventions remain in its historical tables.
Final test seeds were refreshed after initial tests had been viewed; the study was not preregistered.

Use the validated split, artifact, and comparison mechanics. Do not claim repeatable repair benefit,
CAM's incremental value, general bias detection, or cue-invariant logits from these results.

## Controlled checks and decisions

Keep discovery, confirmation, and final evaluation separate. Freeze a shortlist on discovery, then
test on confirmation. In the notebook, both neutralization and training-donor blending must yield
excess predicted-class probability drop >0.02 over matched controls and a descriptive bootstrap lower
bound >0. Area and per-image L1/L2 change match; central label pixels are untouched. These thresholds
are specific to this example. For another task, define its criteria before confirmation and justify them.

Retain every rejected edit, blank map, and extraction error. Record sensitivity to label-preserving
cue changes separately from the location gate: controls can introduce globally pooled color evidence.
Do not promote a proposal to supported because one operator passes or because the top CAM tile
matches a known benchmark cue. Do not declare no shortcut from an unresolved gate.

| Evidence | Next action |
| --- | --- |
| Diagnostic preprocessing differs from trusted evaluation | Fix the diagnostic helper; rerun controls before cue interpretation |
| Only CAM salience or an invalid edit | Report a hypothesis; improve controls or stop unresolved |
| Cue dependence observed, location gate unresolved | Report both; consider an authorized bounded baseline/data experiment, label any targeted arm exploratory |
| Task-preserving checks support the intervention | Run a frozen, budget-matched experiment through the owner's trainer |
| Independent gain with regression or comparator tie | Report all outcomes; do not attribute gain to CAM or call a regressing candidate a verified repair |
| No supported cue dependence | Retain negative evidence and limits; do not prescribe shortcut-specific training |

Check whether authorization already covers local experiments. Continue within it. If a necessary
data or deployment action exceeds it, prepare the concrete proposal and request only that additional
authorization. Investigation permission alone does not authorize production replacement.

## Artifact contracts

Keep TorchCAM's schema-v1 `manifest.json` contract unchanged. It is the completion marker for
per-image maps, heatmaps, overlays, scores, method, layers, image dimensions, and runtime.
Validate it using step 4 of the skill. It does not contain training provenance or repair verification.

The notebook's **separate, unversioned** `experiment.json` contains:

- `protocol`: seeds, regimes, arms, split sizes, epochs, batch, optimizer, centering, tiles,
  edit probability, selection gate, bootstrap, seed offsets, cue agreement, diagnostic budget;
- `runtime`: dependency pins, Python, TorchCAM source revision and version, CPU threads;
- `diagnoses`: regime/seed, `status`, provisional `chosen`, `confirmed`, CAM tile scores,
  all per-image `probes` (`ok`/`blank`/`error`, correctness/error) and tested `interventions`;
- `runs`: regime/seed/arm, weight hash, observed steps/presentations, test predictions,
  average/worst-group accuracy and all four groups with counts;
- `summary`, `test_hashes`, and retained blank/error `fault_checks`.

It exports to a fresh temporary directory and does **not** export checkpoints or schema-v1 bundles.
Capture those separately when adapting it; retain the source revision and hash of the notebook and
JSON. Check split separation, predictions/group counts, budgets, all arms and seeds, finite metrics,
and fault probes. Acceptance checks validate integrity, not improvement.

For an owner's investigation, write a separate JSON record with:

- trusted loader/preprocess/trainer paths and config, checkpoint/class-order/input hashes,
  runtime, original logits, authorization scope;
- discovery/confirmation/evaluation sample IDs or hashes, labels, group counts, and bundle paths;
- frozen hypothesis, edit/control recipes, donors, magnitude/label checks, decision criteria;
- per-sample original/edited scores, correctness, blank/error statuses, supported/unresolved findings;
- experiment arms, seed/checkpoint hashes, observed budgets, selection rules and criteria;
- independent evaluation predictions, average/class/group metrics, seed differences, regressions,
  failed repairs, inference-preservation checks, and limitations.

Use the owner's existing record format if one exists. This record is not a new TorchCAM API schema.

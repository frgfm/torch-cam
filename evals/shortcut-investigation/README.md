# Small paired shortcut-investigation evaluation

This evaluates an agent using the original skill at TorchCAM `5a0bc4d` against a fresh agent using the extended
skill, with the same owner artifacts and budget. It replays [notebooks PR #21](https://github.com/frgfm/notebooks/pull/21)
at `cb719ae40cd137e93cb9a16bdb59f1c03e4ee587`; **no training implementation is duplicated here**.

The prerequisite has no repeatable CAM-guided repair advantage: unchanged, ordinary and provisional guided
continuation all score 100% average/worst-group accuracy across three seeds in both regimes. All location gates
remain unresolved. This evaluation tests investigation, diagnostic fixes and accurate reporting of that null
training outcome, rather than inventing a successful shortcut repair.

## Cases and frozen scoring

| Case | Checkpoint and perturbation | Expected finding and outcome |
| --- | --- | --- |
| Preprocessing | Final no-shortcut unchanged checkpoint; diagnostic helper centers an already centered model input again | Fix helper only; no shortcut training; equal trusted tensors/logits and no group regression on fresh data |
| Genuine shortcut | Four-epoch shortcut pilot; permitted corner red/blue swap | Cue dependence: 100% prediction flips; location gate unresolved; training comparators tie |
| No-shortcut control | Final no-shortcut unchanged checkpoint; same cue swap | No supported shortcut in these checks; no shortcut-specific training; preserve uncertainty |

`prepare.py` verifies the notebook's SHA256, executes its code cells (only removing the display magic), and
captures pilot state dictionaries through a wrapper around its unchanged trainer. It retains the full
`experiment.json`, all final/pilot weights needed for these cases, matched checks, failures/successes, and a
separate label-preserving sensitivity check. The latter does not replace the notebook's matched location gate.
Each condition receives identical copies; only `.agents/skills/torchcam-debug-prediction` differs.

Each fresh agent receives [prompt.txt](prompt.txt), at most **four tool invocations** and a **180-second target**.
Follow-up training may be proposed but cannot be run within this budget. Only the diagnostic helper and new
investigation artifacts may be edited. The synthetic central-label/border-edit authorization does not apply to
owner photographs, data collection, or deployment. The original baseline skill is frozen in [baseline-SKILL.md](baseline-SKILL.md).

The scorer checks diagnosis, unsupported intervention recommendations (training on mismatch/control, supported
guided repair with unresolved confirmation, or a claimed verified training repair), trusted file preservation,
and verified outcomes. A preprocessing repair must produce exactly equal tensors/logits and 100% worst-group
accuracy on 1,024 new examples, generated **after submission** with seed `200000 + case_seed`, 256 per group.
Shortcut/null outcomes must reference the unchanged independent experiment artifact and report zero guided
advantage, no regression, and unresolved location evidence. The control accepts either no repair attempted or
a grounded null experiment report. Missing/invalid submissions and infrastructure/budget failures remain failures.
Integrity hashes live outside agent workspaces so rewriting a local integrity file cannot forge preservation.

## Recorded comparison

The final comparison used case seed **17** on 2026-10-08, after a seed-7 development run exposed confusing
artifact-list iteration. Core scoring was frozen before the final dispatch. The extension makes that iteration
explicit and supplies a bundle validator. The development run and its extraction/validation failures remain in
[results/development-seed7](results/development-seed7); both conditions completed all three records there.

| Final measure | Original skill | Extended skill |
| --- | ---: | ---: |
| Correct diagnosis in final answer (including unsaved answer) | 3/3 | 3/3 |
| Unsupported intervention recommendations in final answer | 0/3 | 0/3 |
| Independent trusted-file integrity check | 3/3 | 3/3 |
| Required response records completed | 2/3 | 3/3 |
| Automatically verified outcome records | 2/3 | 3/3 |
| Actual diagnostic helper repairs independently verified | 1/1 | 1/1 |

Both diagnostic fixes achieve 100% average/worst-group accuracy on 1,024 withheld images; tensors and logits
exactly match trusted inference. The baseline shortcut agent returned the right diagnosis and null repair result,
but its fourth invocation failed while parsing list-valued artifacts; it did not persist `response.json`.
The grader counts that workflow as incomplete, not as an incorrect conceptual diagnosis. Its final answer is
retained separately in [native-final-answer.json](results/seed17/baseline/shortcut/native-final-answer.json).
The baseline control retained two validation errors; the extended cases reported successful failure/control
bundle validation. This single completion difference does not establish a general agent-performance gain.

[scores.json](results/scores.json) scores persisted records only, so its baseline diagnosis/inference counts are
2/3. The table above distinguishes that completion denominator from final-answer correctness and the independent
[integrity audit](results/independent-integrity-audit.json). “Verified outcome” includes correctly reporting no
repair advantage; it does **not** mean a trained shortcut repair was demonstrated.

Raw responses, diagnostic helpers and case evidence are retained under [results/seed17](results/seed17).
[experiment.json.gz](results/experiment.json.gz) is the full notebook run record (including all predictions,
rejected interventions and fault probes), compressed without a timestamp. Its uncompressed SHA256 and source/runtime
pins are in [provenance.json](results/provenance.json). Two complete notebook replays produced byte-identical JSON.

## Reproduce

Use a fresh Linux Python 3.11 environment and the notebook's tested pins. No TorchCAM runtime dependencies change.
Clone `frgfm/notebooks` and check out `cb719ae40cd137e93cb9a16bdb59f1c03e4ee587`, then from the TorchCAM root:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu'
python -m pip install 'numpy==2.4.6' 'matplotlib==3.11.2' 'Pillow==12.3.0' \
  'nbformat==5.10.4' 'nbclient==0.10.2' 'ipykernel==6.30.1' \
  'torchcam @ git+https://github.com/frgfm/torch-cam.git@5a0bc4d439ad642a37ed301d94601048e8d32de8'
MPLBACKEND=Agg python evals/shortcut-investigation/prepare.py \
  /absolute/path/to/notebooks/torch-cam/shortcut_repair.ipynb /tmp/shortcut-eval --case-seed 17
python evals/shortcut-investigation/run.py /tmp/shortcut-eval --python /absolute/path/to/environment/bin/python
python evals/shortcut-investigation/score.py /tmp/shortcut-eval /tmp/shortcut-scores.json
```

The replay uses the trusted notebook's setup check; use a disposable environment because that cell installs pins
if needed. It refuses an unrecognized notebook hash and an existing output directory. Preparation runs the full
18-arm training experiment once per replay, outside the paired agents' budget. `run.py` uses six ephemeral Codex
CLI sessions, unchanged default model/auth, JSONL traces, a 180-second cutoff and interruption on a fifth tool-start
event. It records endpoint/budget failures separately; `score.py` refuses those runs even if a response exists.
Prepare a fresh root for every comparison. Do not reuse repaired helpers or previous submissions.

## Limits and backend details

The recorded run used fresh native collaboration agents (`fork_turns=none`) with the parent's default model and
reasoning effort, without overrides. The exact backend model snapshot was not exposed. All six reported four
invocations. Native wall limits and workspace boundaries were instructed, rather than enforced by OS isolation;
saved final records appeared within 131–169 seconds of round dispatch. The CLI backend above enforces termination
but is a reproduction route, not the measured backend. A local CLI probe (0.159.0-alpha.3) was blocked by the
environment proxy with HTTP CONNECT 403 at the model endpoint; no CLI model score is claimed. Do not bypass that
policy or use another identity to make it run.

This is one final agent attempt per condition/case, plus a disclosed development run. It is an artifact-rich,
synthetic regression check with supplied label-preserving edits, not blinded generalization testing, a statistical
performance estimate, or evidence that the skill improves training. Agents read existing outcomes rather than
training new repairs; final candidate weights are not included in the notebook JSON. The actual independently
verified change is diagnostic preprocessing. The training outcome remains null, with historical failed attempts
retained. Score comparisons after future edits need new fresh sessions and must retain all failures.

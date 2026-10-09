# Paired shortcut-investigation evaluation

The goal is a reproducible investigation that leads to a justified next step under a fixed budget.
Correct labels alone do not establish that value. Reuse the hash-pinned [notebooks PR #21](https://github.com/frgfm/notebooks/pull/21)
trainer, checkpoints and artifacts; no training implementation is copied.

The rubric is fixed before the blind comparison:

| Measure | Success criterion |
| --- | --- |
| **Evidence-backed resolution** (primary, all cases) | Correct diagnosis; all supplied-checkpoint measurements reproduce; grounded decision; preserved inference; within budget; no unsupported intervention or false repair claim |
| Diagnostic fix (mismatch cases only) | Editable helper matches owner tensors/logits and reaches 100% worst-group accuracy on 1,024 fresh images |
| Unsupported interventions / false repair claims | Count each separately; lower is better |
| Model repair advantage | Independent worst-group gain over unchanged **and** ordinary continuation, with frozen regression tolerances; a tie is no advantage |

The primary measure checks inference parity before edits, cue-swap flip rate and probability change,
CAM coverage of successes/failures including unusable maps, and matched-control excess/intervals.
Numeric tolerances are absolute 1e-5/relative 1e-4. Raw data and opaque case identifiers replace the former
answer-bearing evidence summaries. The expected measurements and case map remain outside agent workspaces.
Both conditions receive the same artifact-list schema clarification, isolating investigation guidance
from a known baseline parsing error. Missing records fail the primary measure; safety rates use completed
submissions as their denominator, since an absent response is not a demonstrated safe recommendation.
Training is proposed, not rerun by participants; artifact-backed null decisions are measured separately
from diagnostic fixes and never counted as model repairs.

Use the notebook's pinned Linux Python 3.11 environment and a fresh output directory:

```sh
MPLBACKEND=Agg python evals/shortcut-investigation/prepare.py /path/to/shortcut_repair.ipynb /tmp/shortcut-eval --case-seed 27
bash evals/shortcut-investigation/run.sh /tmp/shortcut-eval /path/to/environment/bin/python
python evals/shortcut-investigation/score.py /tmp/shortcut-eval /tmp/shortcut-scores.json
```

Preparation checks the notebook SHA256, replays its code (removing only display magic), captures pilot
weights around the unchanged trainer, and records trusted hashes outside agent workspaces. Install its exact
dependency/source pins in a disposable environment first; mismatched imports require a fresh process.
Cases are double-centering, a cue-dependent four-epoch pilot, and a trained no-shortcut control.
The original skill comes from TorchCAM `5a0bc4d439ad642a37ed301d94601048e8d32de8`.
Each condition receives identical owner inputs, the same [prompt](prompt.txt), four tool invocations and
180 seconds; only the skill differs. The Bash runner requires jq, GNU timeout and setsid, retains traces,
and rejects endpoint/budget failures. Never reuse repaired workspaces. Missing/invalid submissions remain
failures in the denominator. Keep generated data, weights and results outside the repository.

The previous answer-supplied seed-17 comparison gave both conditions 3/3 correct final-answer diagnoses,
zero unsupported recommendations and one verified diagnostic fix. Persisted records were 2/3 versus 3/3;
that does **not** establish improved investigation or repair. Its `verified_outcome` metric mixed fixes
with correct null reports and has been removed. Seed-7 development records completed 3/3 in both conditions.

All 18 final notebook outcomes tie at 100% average/worst-group accuracy across seeds 7/17/27:
**no CAM repair advantage**; all six pilot location gates remain unresolved. Two notebook replays matched.
The accuracy ceiling cannot establish training repair value. No owner data changes or deployment are
part of this evaluation. See the skill's investigation contract for study limits.

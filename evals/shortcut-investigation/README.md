# Paired shortcut-investigation evaluation

The goal is a reproducible investigation leading to a justified next step under a fixed budget.
The corrected blind seed-17 comparison shows **no measured agent advantage**:

| Measure | Original | Extended |
| --- | ---: | ---: |
| Evidence-backed investigations (provisional) | 3/3 | 3/3 |
| Correct diagnosis / reproducible measurements | 3/3 / 3/3 | 3/3 / 3/3 |
| Inference preserved on fresh data | 3/3 | 3/3 |
| Diagnostic fix: 1,024 fresh images, worst-group accuracy | 1/1; 100% | 1/1; 100% |
| Unsupported interventions / false repair claims | 0/3 / 0/3 | 0/3 / 0/3 |
| Observed budget verification | unavailable | unavailable |

**Primary criterion:** correct diagnosis, independently matching measurements, grounded decision, preserved
inference, no unsupported intervention or false repair claim, under the common budget. Label accuracy alone
is insufficient. Measurements check pre-edit inference parity, cue-swap flip rate/probability change,
CAM successes/failures including unusable maps, and matched-control excess/intervals. Numeric tolerances
are absolute 1e-5/relative 1e-4. Diagnostic fixes additionally require exact owner tensors/logits and 100%
worst-group accuracy on untouched data. Null training decisions are not successful repairs.

**Training value:** independent worst-group gain over unchanged **and** ordinary continuation, subject to
frozen regression tolerances. The reused [notebooks PR #21](https://github.com/frgfm/notebooks/pull/21)
outcomes tie at 100% average/worst-group accuracy in all 18 arms across seeds 7/17/27: **0 percentage-point
CAM advantage**. All six pilot location gates remain unresolved. The accuracy ceiling cannot establish
training repair value; this workflow remains exploratory.

Use the notebook's pinned Linux Python 3.11 environment and a fresh output directory:

```sh
MPLBACKEND=Agg python evals/shortcut-investigation/prepare.py /path/to/shortcut_repair.ipynb /tmp/shortcut-eval --case-seed 17
bash evals/shortcut-investigation/run.sh /tmp/shortcut-eval /path/to/environment/bin/python
python evals/shortcut-investigation/score.py /tmp/shortcut-eval /tmp/shortcut-scores.json
```

Preparation checks the notebook SHA256, replays its code (removing only display magic), and captures pilot
weights around the unchanged trainer; no training implementation is copied. Install its exact dependency/source
pins in a disposable environment first; mismatched imports require a fresh process. Cases are double-centering,
a cue-dependent four-epoch pilot, and a trained no-shortcut control. Raw public views and opaque identifiers
replace supplied diagnostic answers; expected measurements, case mappings and trusted hashes stay outside
agent workspaces. Generated data, weights and results stay outside the repository.

The original skill comes from TorchCAM `5a0bc4d439ad642a37ed301d94601048e8d32de8`; the corrected trial used
[extended revision 745c936](https://github.com/frgfm/torch-cam/tree/745c936/.agents/skills/torchcam-debug-prediction).
Conditions receive identical owner inputs, the same [prompt](prompt.txt), four tool invocations and 180 seconds;
only the skill differs. Both get the artifact-list schema clarification. Training is proposed, not rerun by
participants. The Bash runner requires jq, GNU timeout and setsid, retains traces, and rejects endpoint/budget
failures. Never reuse repaired workspaces. Missing records fail the primary measure; safety denominators use
valid submissions, since absence does not demonstrate a safe recommendation.

The initial blind seed-27 round completed 1/3 versus 2/3 investigations: two original-skill artifact-parsing
failures and one extended-skill source-inspection failure. The repeat shared the schema clarification and added
notebook cell-source/record-preservation guidance to the skill. Review then removed two unpublished grading
constraints for **both** conditions: string-only observations and treating ambiguous `unchanged` as a
shortcut-specific intervention. Structured observations are accepted; raw recommendations remain unchanged.
Unsupported interventions mean shortcut-specific changes without cue evidence or guided changes claimed as
supported while confirmation is unresolved. The earlier answer-supplied comparison's mixed `verified_outcome`
metric was removed. These small trials do not demonstrate repeatable incremental investigation value.

Fresh native agents used inherited defaults; the backend snapshot was unavailable. All reported four calls;
time and workspace limits were instructed, not isolated. The extended mismatch agent's final text was
unconfirmed while its command completed; its saved record independently passes. `budget_verified` requires
runner execution metadata: native investigation scores are provisional, not verified primary-budget successes.
The CLI route was proxy-blocked (CONNECT 403), so no enforced-budget comparison is claimed. No owner data changes
or deployment occur. See the skill's investigation contract for notebook study limits.

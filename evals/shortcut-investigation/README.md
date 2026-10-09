# Paired shortcut-investigation evaluation

Reuse the hash-pinned [notebooks PR #21](https://github.com/frgfm/notebooks/pull/21) trainer and artifacts:
no training implementation is copied. Cases are a double-centering diagnostic helper, a genuinely cue-dependent
four-epoch pilot, and a trained no-shortcut control. Only the installed skill differs between conditions.

Use the notebook's pinned Linux Python 3.11 environment and a fresh output directory:

```sh
MPLBACKEND=Agg python evals/shortcut-investigation/prepare.py /path/to/shortcut_repair.ipynb /tmp/shortcut-eval
bash evals/shortcut-investigation/run.sh /tmp/shortcut-eval /path/to/environment/bin/python
python evals/shortcut-investigation/score.py /tmp/shortcut-eval /tmp/shortcut-scores.json
```

Preparation checks the notebook SHA256, replays its code (removing only display magic), captures pilot weights
around the unchanged trainer, and records trusted hashes outside agent workspaces. Install the notebook's exact
dependency/source pins in a disposable environment before preparation; importing mismatched pins requires a
fresh process. The original skill comes from TorchCAM commit `5a0bc4d439ad642a37ed301d94601048e8d32de8`.
Each agent gets the same [prompt](prompt.txt), four tool invocations and a 180-second target. Only diagnostics
and new investigation artifacts may change; training is proposed, not run. The CLI runner retains traces and
rejects endpoint/budget failures; it requires Bash, jq, GNU timeout and setsid. Never reuse repaired workspaces.

The scorer checks diagnosis, unsupported recommendations, trusted-file integrity and outcomes. A diagnostic
fix must exactly reproduce trusted tensors/logits and score 100% worst-group on 1,024 images generated after
submission (256/group). Null training outcomes require the independent artifact, comparator ties, no regression
and unresolved location evidence. Missing/invalid submissions stay failures; “verified outcome” includes honest
null reporting and does not establish a trained repair.

| Final seed-17 measure | Original | Extended |
| --- | ---: | ---: |
| Final-answer diagnosis / unsupported recommendations | 3/3 / 0/3 | 3/3 / 0/3 |
| Independent trusted-file integrity | 3/3 | 3/3 |
| Persisted, verified outcome records | 2/3 | 3/3 |
| Diagnostic fixes verified on 1,024 fresh images | 1/1 | 1/1 |

Both diagnostic fixes reach 100% average/worst-group accuracy. The baseline shortcut answer was correct but
artifact-list parsing exhausted its budget before saving `response.json`; persisted-record scorer counts are
therefore 2/3. Both training comparators tie guided training: **no CAM repair advantage**. Two full notebook
replays produced identical JSON. The seed-7 development round, failed completion and raw submissions remain in
[results.json.gz](results.json.gz), alongside one full notebook record, provenance, checks and integrity audit.
Inspect with `gzip -dc evals/shortcut-investigation/results.json.gz`; `measured_revision` pins the evaluated
skill and harness. Repeated evidence already present in the notebook record is referenced by regime/seed.

Measurements used fresh native agents with inherited defaults; the exact backend snapshot was unavailable.
All reported four invocations; native time/boundary limits were instructed, not isolated. Saved records arrived
within 131–169 seconds. The CLI route is not the measured backend: its local probe was proxy-blocked (CONNECT
403). One final attempt per condition/case after a disclosed development run, supplied synthetic edits and
an accuracy ceiling do not establish general agent improvement or training repair value. No owner data changes
or deployment are authorized by this evaluation. See the skill's investigation contract for study limits.

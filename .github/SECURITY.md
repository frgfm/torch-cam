# Reporting security issues

If you believe you have found a security vulnerability in TorchCAM, we encourage you to let us know right away. We will investigate all legitimate reports and do our best to quickly fix the problem.

Please report security issues using https://github.com/frgfm/torch-cam/security/advisories/new

## Dependency and deployment checks

Load only trusted models. Immutable model revisions and clean advisory scans do not establish runtime safety.

Demo dependencies are declared in `pyproject.toml`'s `demo` extra. CI installs the built wheel with that extra;
the Space installs it from an immutable TorchCAM source revision. Transitive versions resolve at install time.
The Space owns its Dockerfile and OS maintenance; passing CI does not verify deployment.

One Streamlit process admits one active computation; concurrent requests receive a busy message without queuing.
Upload, image, CAM and generation limits are enforced in `demo/app.py` and `demo/inference.py`.
`make run-demo` and Space synchronization set Streamlit's server transport limits before startup.
Work keeps its slot until it returns or raises. Hard cancellation or more workers require process isolation
and shared resource budgets. Total sessions and process memory are not capped.

Offline check: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python demo/smoke.py`.

Optional model checks run on the default branch with `pretrained: true`. An owner must first configure
**HF_MODEL_READ_TOKEN** with model-read access and accept Gemma's conditions; missing credentials fail before downloads.
Restrict existing **HF_TOKEN** to writing `spaces/frgfm/torch-cam`; use it only in the separate synchronization job.
Offline checks need no token.

## Owner settings proposal

Apply separately; this work changes no repository settings.

| Setting | Proposed action |
| --- | --- |
| Actions default workflow permissions | Set read-only; jobs request needed writes. |
| Actions PR creation and approval | Disable after replacing the annual job's PR-creation path. |
| Secret scanning and push protection | Enable and review alerts. |
| Secret validity checks / extra patterns | Enable where available and review false positives. |
| HF model-read secret | Add the scoped secret above only for trusted pretrained checks. |

GitHub combines workflow-token PR creation and approval in one control. Disabling it also blocks
the annual copyright job's `GITHUB_TOKEN` PR creation; coordinate that owner change first.

Protect deploy and package-publish credentials with environments restricted to trusted branches/tags and
required owner approval where available. Keep PyPI trusted publishing; review unused legacy credentials
before removing them. Review code-scanning coverage
in the Security UI: a missing API result or zero dependency alerts is not evidence of code, container, model
or deployed-runtime scanning. Configure missing supported coverage separately.

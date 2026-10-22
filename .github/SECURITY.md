# Reporting security issues

If you believe you have found a security vulnerability in TorchCAM, we encourage you to let us know right away. We will investigate all legitimate reports and do our best to quickly fix the problem.

Please report security issues using https://github.com/frgfm/torch-cam/security/advisories/new

## Dependency and deployment checks

Load only trusted models. Immutable model revisions and clean advisory scans do not establish runtime safety.

`demo/requirements.txt` records the Linux x86_64, Python 3.13 CPU environment. Refresh it with `make lock-demo`
then run `make check-demo-lock` and the offline check below. CI and the Space install that resolution;
Space requirements also pin TorchCAM's source revision. The Space owns its Dockerfile and OS maintenance.
`make install-demo` is a development install; passing CI does not verify deployment.

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

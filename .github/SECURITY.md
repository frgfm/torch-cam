# Reporting security issues

If you believe you have found a security vulnerability in TorchCAM, we encourage you to let us know right away. We will investigate all legitimate reports and do our best to quickly fix the problem.

Please report security issues using https://github.com/frgfm/torch-cam/security/advisories/new

## Dependency and deployment checks

Load only trusted models. The public demo also parses uploads and downloads allow-listed models;
immutable revisions and clean advisory scans do not establish runtime safety.

`demo/requirements.txt` records the Linux x86_64, Python 3.13 CPU environment. Refresh it with `make lock-demo`
using uv 0.13.4, then run `make deps-check` and the offline checks below. CI and Space use the same resolution,
Dockerfile and immutable TorchCAM source revision. `make install-demo` is a development install.
Maintain OS packages, models and the host separately; a source or image test does not verify deployment.

One Streamlit process admits one active computation and one waiter, with a 15-second queue timeout.
Uploads are limited to 8 MiB and 8 million pixels. CAM limits are four layers, batch size 32 and eight
FinerCAM comparison classes; generation limits are 24 new BLIP tokens or 48 Qwen/Gemma tokens.
Work retains its slot until it returns or raises. Hard CPU/GPU cancellation requires process isolation;
additional workers require shared admission and memory budgeting. Total sessions and process memory are not capped.

Scheduled audits check the unchanged demo lock and fresh current/minimum core/test resolutions, without
advisory exclusions. Only advisory lookup strips PyTorch's `+cpu` suffix; any skipped package fails the audit.
Update and re-audit dependencies when new advisories appear.

Offline validation needs no Hugging Face credential:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --no-sync python demo/smoke.py
```

Optional model checks run on the default branch with `pretrained: true`. An owner must first configure
**HF_MODEL_READ_TOKEN** with read-only access to the listed models and accept Gemma's access conditions;
missing credentials fail before downloads. Restrict existing **HF_TOKEN** to writing `spaces/frgfm/torch-cam`.
Use it only for Space synchronization, never model checks or Space runtime. Offline checks need no token.

# Reporting security issues

If you believe you have found a security vulnerability in TorchCAM, we encourage you to let us know right away. We will investigate all legitimate reports and do our best to quickly fix the problem.

Please report security issues using https://github.com/frgfm/torch-cam/security/advisories/new

## Dependency and deployment checks

The library consumes caller-provided models and tensors. The public demo also parses uploads, downloads
allow-listed models, and runs expensive inference. A clean advisory scan does not establish that these paths
are safe. Only load trusted model files; immutable revisions identify reviewed inputs, not a security sandbox.

`demo/requirements.txt` records the Linux x86_64, Python 3.13 CPU environment. Refresh it with `make lock-demo`
using uv 0.13.4, then run `make deps-check` and the offline demo checks. CI installs this file and the built
TorchCAM wheel. Space synchronization copies this same resolution and `demo/Dockerfile`, plus an immutable
TorchCAM source-revision requirement. Local `make install-demo` remains a development install.
The lock pins Python packages; the base image, operating-system packages, model files and host need separate
maintenance. No Space deployment is verified by a source or wheel test.

The single Streamlit process admits one active computation and at most one queued request. Queue waiting
expires after 15 seconds. Uploads are limited to 8 MiB and 8 million decoded pixels before pixel loading.
CAM requests use at most four layers, masked-input batches of at most 32, and at most eight FinerCAM
comparison classes. Generation uses at most 24 new tokens for BLIP and 48 for Qwen/Gemma. Existing work
keeps its slot until it returns or raises, including session cancellation. There is no hard in-flight
CPU/GPU deadline; terminating native inference requires process isolation. Extra server workers would each
have their own gate and must not be added without shared admission and memory budgeting.

The dependency-audit workflow rechecks the unchanged demo resolution and fresh current/minimum core/test
resolutions on a schedule. No advisory exclusions are used. Compatibility floors are not a promise against
future disclosures; update and re-audit the resolved environment when an advisory appears.
The audit maps PyTorch's `+cpu` wheel suffix to the corresponding public release for advisory lookup and
fails if any package is skipped. Installed versions keep the CPU suffix.

Offline validation needs no Hugging Face credential:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --no-sync python demo/smoke.py
```

Model downloads are a separate, optional `demo` workflow-dispatch check (`pretrained: true`). Before using it,
an owner must configure **HF_MODEL_READ_TOKEN**, a fine-grained token with read access only to the listed
model repositories, and accept the Gemma model's access conditions. The check fails before downloads when
this secret is absent. Never reuse the Space-write token. The existing **HF_TOKEN** is used only by the
Space synchronization step; restrict it to write access to `spaces/frgfm/torch-cam`. Do not configure a
Space-write token as a runtime Space secret. No new token is needed for public/offline checks.

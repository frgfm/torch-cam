"""Run the PR's unchanged DEX-AR example on a genuine pretrained Qwen checkpoint."""

import argparse
from contextlib import contextmanager
import base64
import hashlib
import json
import os
from pathlib import Path
import resource
from urllib.request import urlretrieve

import numpy as np
import torch

from scripts import dexar_example

output = Path("dexar-demo-results")
output.mkdir(exist_ok=True)
image_source = "https://raw.githubusercontent.com/huggingface/transformers/main/tests/fixtures/tests_samples/COCO/000000039769.png"
image_path = output / "input.png"
urlretrieve(image_source, image_path)
input_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
revision = "66285546d2b821cf421d4f5eb2576359d3770cd3"
torch.manual_seed(0)
torch.set_num_threads(4)
original_save = dexar_example.save_overlays
original_timer = dexar_example.timer

@contextmanager
def logged_timer(timings, key, device):
    with original_timer(timings, key, device):
        yield
    print(f"Completed {key}: cumulative {timings[key]:.3f}s", flush=True)

dexar_example.timer = logged_timer

def save_arrays(image, answer, tokens, maps, weights, sequence, output_dir):
    np.savez_compressed(
        output_dir / "maps.npz",
        token_maps=maps.cpu().numpy(),
        token_weights=weights.cpu().numpy(),
        sequence_map=sequence.cpu().numpy(),
    )
    original_save(image, answer, tokens, maps, weights, sequence, output_dir)

dexar_example.save_overlays = save_arrays
print(f"Pretrained model: {model_id} @ {revision}", flush=True)
dexar_example.main(argparse.Namespace(
    image=str(image_path),
    output=str(output),
    model=model_id,
    revision=revision,
    device="cpu",
    dtype="float32",
    prompt="Describe this image in one sentence.",
    max_pixels=256 * 28 * 28,
    max_new_tokens=24,
))
report = json.loads((output / "report.json").read_text())
report.update(
    pretrained=True,
    input_source=image_source,
    input_sha256=input_sha256,
    seed=0,
    cpu_count=os.cpu_count(),
    threads=torch.get_num_threads(),
    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    implementation_commit="ea636ab2bb38443a2fb2730d410ac19f7c2ea48b",
)
(output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print("DEXAR_REPORT_JSON:" + json.dumps(report, separators=(",", ":")), flush=True)
print("DEXAR_MAPS_BASE64:" + base64.b64encode((output / "maps.npz").read_bytes()).decode(), flush=True)

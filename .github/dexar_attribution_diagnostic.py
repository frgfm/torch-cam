"""One-image diagnostic; kept outside the implementation PR."""
import argparse
import base64
import hashlib
import io
import json
import math
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from scripts import dexar_example as example
from torchcam.methods import DEXAR, TAM

ROOT = Path('dexar-diagnostic-results')
ROOT.mkdir(exist_ok=True)
MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'
REVISION = '66285546d2b821cf421d4f5eb2576359d3770cd3'
URL = 'https://raw.githubusercontent.com/huggingface/transformers/main/tests/fixtures/tests_samples/COCO/000000039769.png'
data = urllib.request.urlopen(URL).read()
(ROOT / 'input.png').write_bytes(data)
image = Image.open(io.BytesIO(data)).convert('RGB')
torch.manual_seed(0)
torch.set_num_threads(4)
processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, min_pixels=4*28*28, max_pixels=256*28*28)
captured = {}
layer_maps, visual_max, text_max, logits, rms = [], [], [], [], []
tam_maps = {1: [], 3: []}
original_call = DEXAR.__call__
original_explain = example.explain_qwen
original_timer = example.timer

@contextmanager
def logged_timer(timings, key, device):
    with original_timer(timings, key, device):
        yield
    print('PHASE', key, round(timings[key], 3), flush=True)

def record_gradients(self, scores, attentions, mask, **kwargs):
    rows = []
    original_grad = torch.autograd.grad
    def record(*args, **options):
        result = original_grad(*args, **options)
        rows.append(result[0][:, :, -1].detach().float().cpu().relu())
        return result
    torch.autograd.grad = record
    try:
        result = original_call(self, scores, attentions, mask, **kwargs)
    finally:
        torch.autograd.grad = original_grad
    rows = torch.stack(rows)
    visual = rows[..., mask.cpu()]
    vmax, tmax = visual.amax(-1), rows[..., ~mask.cpu()].amax(-1)
    raw = (visual * (vmax-tmax).relu().unsqueeze(-1)).sum(2)
    normalized = DEXAR._normalize(raw.sum(0).reshape(1, *self.grid_shape))
    torch.testing.assert_close(normalized, result[0].cpu())
    torch.testing.assert_close((vmax.amax((0, 2))-tmax.amax((0, 2))).relu(), result[1].cpu())
    layer_maps.append(raw[:, 0].reshape(-1, *self.grid_shape).numpy())
    visual_max.append(vmax[:, 0].numpy())
    text_max.append(tmax[:, 0].numpy())
    logits.append(torch.stack(scores).detach()[:, 0].cpu().numpy())
    return result

def record_model(model, inputs, answer_ids, **kwargs):
    captured.update(model=model, inputs=inputs, answer_ids=answer_ids)
    grid = example.visual_grid(model, inputs)
    prompt_ids = inputs['input_ids']
    visual_mask = prompt_ids[0] == model.config.image_token_id
    text_mask = ~visual_mask
    for special in processor.tokenizer.all_special_ids:
        text_mask &= prompt_ids[0] != special
    context_ids = prompt_ids[:, text_mask].clone()
    context = None
    visual = None
    step = 0
    methods = {k: TAM(model.get_output_embeddings(), kernel_size=k) for k in tam_maps}
    def observe(_model, _args, output):
        nonlocal step, context, visual, context_ids
        states = output.hidden_states[-1].detach()
        if step == 0:
            visual = states[:, visual_mask].reshape(1, *grid, states.shape[-1])
            context = states[:, text_mask]
        token = int(answer_ids[0, step])
        for k, method in methods.items():
            tam_maps[k].append(method(token, visual, context, context_ids).cpu().float().numpy())
        rms.append(np.array([float(s[:, -1].detach().float().square().mean().sqrt()) for s in output.hidden_states[1:]]))
        context = torch.cat((context, states[:, -1:]), dim=1)
        context_ids = torch.cat((context_ids, answer_ids[:, step:step+1]), dim=1)
        step += 1
    handle = model.register_forward_hook(observe)
    try:
        result = original_explain(model, inputs, answer_ids, **kwargs)
    finally:
        handle.remove()
    captured['result'] = result
    return result

DEXAR.__call__ = record_gradients
example.explain_qwen = record_model
example.timer = logged_timer
args = argparse.Namespace(image=str(ROOT/'input.png'), output=str(ROOT), model=MODEL, revision=REVISION,
                          device='cpu', dtype='float32', prompt='Describe this image in one sentence.',
                          max_pixels=256*28*28, max_new_tokens=24)
example.main(args)
DEXAR.__call__ = original_call
example.explain_qwen = original_explain
example.timer = original_timer
maps, weights, sequence, _ = captured['result']
raw = torch.from_numpy(np.stack(layer_maps))
vmax = torch.from_numpy(np.stack(visual_max))
tmax = torch.from_numpy(np.stack(text_max))
extractor = DEXAR(tuple(maps.shape[-2:]))
arrays = dict(token_maps=maps.cpu().numpy(), token_weights=weights.cpu().numpy(), sequence_map=sequence.cpu().numpy(),
              layer_maps=raw.numpy(), visual_max=vmax.numpy(), text_max=tmax.numpy(), layer_logits=np.stack(logits),
              hidden_rms=np.stack(rms))
for name, start in [('all', 0), ('last10', raw.shape[1]-10)]:
    token_raw = raw[:, start:].sum(1).unsqueeze(0)
    token = DEXAR._normalize(token_raw)
    weight = (vmax[:, start:].amax((1, 2))-tmax[:, start:].amax((1, 2))).relu().unsqueeze(0)
    arrays[name+'_maps'] = token.numpy()
    arrays[name+'_weights'] = weight.numpy()
    arrays[name+'_sequence'] = extractor.aggregate(token, weight).numpy()
    arrays[name+'_raw_sequence'] = DEXAR._normalize((token_raw*weight[..., None, None]).sum(1)).numpy()
for k, values in tam_maps.items():
    arrays['tam'+str(k)+'_maps'] = np.stack(values, axis=1)
np.savez_compressed(ROOT/'diagnostic.npz', **arrays)

# A small conditional-confidence deletion check; actual generated IDs remain fixed.
model, inputs, answer_ids = (captured[k] for k in ['model', 'inputs', 'answer_ids'])
messages = [{'role':'user','content':[{'type':'image'},{'type':'text','text':args.prompt}]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
head = model.get_output_embeddings()
handle = head.register_forward_pre_hook(lambda _m, a: (a[0][:, -1:] if a[0].ndim == 3 else a[0],))
def log_probability(picture, step):
    batch = processor(text=[prompt], images=[picture], return_tensors='pt')
    torch.testing.assert_close(batch['input_ids'], inputs['input_ids'])
    torch.testing.assert_close(batch['image_grid_thw'], inputs['image_grid_thw'])
    batch['input_ids'] = torch.cat((batch['input_ids'], answer_ids[:, :step]), dim=1)
    batch['attention_mask'] = torch.ones_like(batch['input_ids'])
    with torch.no_grad():
        output = model(**batch, use_cache=False, output_attentions=False, output_hidden_states=False)
        return float(output.logits[0, -1].float().log_softmax(-1)[int(answer_ids[0, step])])
def delete_cells(score, largest=True):
    flat = score.flatten()
    count = math.ceil(flat.size*.2)
    selected = np.argsort(flat)[-count:] if largest else np.argsort(flat)[:count]
    mask = np.zeros_like(flat, dtype=np.uint8)
    mask[selected] = 255
    mask = np.asarray(Image.fromarray(mask.reshape(score.shape)).resize(image.size, Image.Resampling.NEAREST)) > 0
    pixels = np.array(image)
    pixels[mask] = np.round(pixels.mean((0, 1))).astype(np.uint8)
    return Image.fromarray(pixels), int(mask.sum())
deletions = []
try:
    for step in [1, 7]:
        baseline = log_probability(image, step)
        candidates = [('all', arrays['all_maps'][0, step], True), ('last10', arrays['last10_maps'][0, step], True),
                      ('tam3', arrays['tam3_maps'][0, step], True), ('all_bottom', arrays['all_maps'][0, step], False),
                      ('random', np.random.default_rng(0).random(maps.shape[-2:]), True)]
        for name, score, largest in candidates:
            picture, removed = delete_cells(score, largest)
            probability = log_probability(picture, step)
            result = dict(position=step+1, token=processor.tokenizer.decode([int(answer_ids[0, step])]),
                          method=name, baseline_log_probability=baseline, deleted_log_probability=probability,
                          delta_nll=baseline-probability, deleted_pixels=removed)
            deletions.append(result)
            print('DELETION', json.dumps(result), flush=True)
finally:
    handle.remove()
report = json.loads((ROOT/'report.json').read_text())
report.update(input_sha256=hashlib.sha256(data).hexdigest(), deletion_fraction_cells=.2, deletions=deletions,
              note='One image; layer ablation and conditional-confidence deletion diagnostic, not an accuracy benchmark.')
(ROOT/'diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
print('DEXAR_DIAGNOSTIC_JSON:'+json.dumps(report, separators=(',', ':')), flush=True)
encoded = base64.b64encode((ROOT/'diagnostic.npz').read_bytes()).decode('ascii')
for offset in range(0, len(encoded), 4096):
    print('DEXAR_DIAGNOSTIC_CHUNK:'+str(offset//4096)+':'+encoded[offset:offset+4096], flush=True)

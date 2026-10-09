"""Inspect and render the diagnostic's genuine pretrained maps without contrast tricks."""
import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchcam.methods import DEXAR

root = Path('dexar-diagnostic-results')
report = json.loads((root/'diagnostic.json').read_text())
arrays = np.load(root/'diagnostic.npz', allow_pickle=False)
arrays = {name: arrays[name].copy() for name in arrays.files}
# Reconstruct the reference variant from preserved raw layer maps.
# TorchCAM's normalization mutates its input, so normalize a separate tensor.
for name, start in [('all', 0), ('last10', 26)]:
    raw = torch.from_numpy(arrays['layer_maps'][:, start:].sum(1).copy()).unsqueeze(0)
    weight = torch.from_numpy(arrays[name+'_weights'])
    arrays[name+'_raw_sequence'] = DEXAR._normalize((raw*weight[..., None, None]).sum(1)).numpy()
np.savez_compressed(root/'diagnostic.npz', **arrays)
original = dict(token_maps=arrays['all_maps'], token_weights=arrays['all_weights'], sequence_map=arrays['all_sequence'])
assert report['revision'] == '66285546d2b821cf421d4f5eb2576359d3770cd3'
assert report['answer'] == 'Two cats are laying on a pink couch with remote controls nearby.'
for name in ['token_maps', 'token_weights', 'sequence_map']:
    np.testing.assert_allclose(arrays[name], original[name], rtol=2e-5, atol=2e-6)
for name in ['all', 'last10']:
    extractor = DEXAR(tuple(report['grid_shape']))
    result = extractor.aggregate(torch.from_numpy(arrays[name+'_maps']), torch.from_numpy(arrays[name+'_weights']))
    torch.testing.assert_close(result, torch.from_numpy(arrays[name+'_sequence']))
image = Image.open(root/'input.png').convert('RGB')
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout='constrained')
methods = [('all_maps', 'DEX-AR: all language layers'), ('last10_maps', 'DEX-AR: last ten layers'), ('tam3_maps', 'TorchCAM TAM: default filter')]
for row, step in enumerate([1, 7]):
    for col, (key, title) in enumerate(methods):
        axis = axes[row, col]
        axis.imshow(image)
        heatmap = torch.from_numpy(arrays[key][0, step])
        resized = F.interpolate(heatmap[None, None], image.size[::-1], mode='bilinear', align_corners=False)[0, 0]
        colored = axis.imshow(resized.numpy(), cmap='turbo', vmin=0, vmax=1, alpha=.45)
        axis.set_title(title if row == 0 else '')
        axis.text(.01, .025, f"Token {step+1}: {report['token_text'][step]!r} · ID {report['explained_ids'][step]}",
                  transform=axis.transAxes, color='white', fontsize=10,
                  bbox=dict(facecolor='black', alpha=.7, edgecolor='none'))
        axis.axis('off')
fig.colorbar(colored, ax=axes.ravel().tolist(), fraction=.015, pad=.01, ticks=[0, 1], label='Relative relevance within each map')
fig.suptitle('Qwen2.5-VL-3B-Instruct · actual answer: '+textwrap.fill(report['answer'], 100)+'\nSame 13×18 grid, image, generated IDs and rendering', fontsize=12)
fig.savefig(root/'comparison.png', dpi=160, facecolor='white')
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
for axis, key, title in zip(axes, ['sequence_map','all_raw_sequence','last10_sequence'],
                            ['Paper aggregation · all layers','Reference aggregation · all layers','Paper aggregation · last ten layers']):
    axis.imshow(image)
    heatmap = torch.from_numpy(arrays[key][0])
    resized = F.interpolate(heatmap[None,None], image.size[::-1], mode='bilinear', align_corners=False)[0,0]
    axis.imshow(resized.numpy(), cmap='turbo', vmin=0, vmax=1, alpha=.45)
    axis.set_title(title, fontsize=10)
    axis.axis('off')
fig.suptitle('Same actual answer; changing aggregation does not produce clean object boundaries')
fig.savefig(root/'sequence-comparison.png', dpi=160, facecolor='white')
plt.close(fig)

layer_mass = arrays['layer_maps'].sum((-2, -1))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), layout='constrained')
summary = []
for axis, step in zip(axes, [1, 7]):
    mass = layer_mass[step]
    share = mass/mass.sum()
    axis.bar(np.arange(1, len(mass)+1), share*100, color=['#3267ab']*26+['#d36e2d']*10)
    axis.set(title=report['token_text'][step].strip(), xlabel='Language layer', ylabel='Share of raw filtered gradient mass (%)')
    axis.set_xticks([1, 6, 12, 18, 24, 30, 36])
    summary.append(dict(token=report['token_text'][step], first_layer_mass_share=float(share[0]),
                        first26_mass_share=float(share[:26].sum()), last10_mass_share=float(share[-10:].sum()),
                        all_weight=float(arrays['all_weights'][0, step]), last10_weight=float(arrays['last10_weights'][0, step]),
                        all_last10_map_correlation=float(np.corrcoef(arrays['all_maps'][0,step].ravel(), arrays['last10_maps'][0,step].ravel())[0,1])))
fig.suptitle('Which layers dominate? Orange bars are the reference wrapper’s last ten layers')
fig.savefig(root/'layer-contributions.png', dpi=160, facecolor='white')
plt.close(fig)
report['layer_summary'] = summary
report['timing_note'] = 'Diagnostic attribution includes TAM and recording overhead; use the original uninstrumented run for attribution cost.'
report['source_run'] = 38000575217
report['raw_sequence_note'] = 'Reconstructed from unchanged raw per-layer maps; cloning preserves raw amplitude before normalization.'
for name in ['all', 'last10']:
    report[name+'_paper_reference_sequence_correlation'] = float(np.corrcoef(arrays[name+'_sequence'].ravel(), arrays[name+'_raw_sequence'].ravel())[0,1])
(root/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({key:report[key] for key in ['answer','layer_summary','deletions','all_paper_reference_sequence_correlation','last10_paper_reference_sequence_correlation']},indent=2))

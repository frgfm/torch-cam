"""Render all predeclared review cases, preserving raw maps and original generated targets."""
import json
from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent
summary = []
for directory in sorted(ROOT.iterdir()):
    if not directory.is_dir() or not (directory/'report.json').exists():
        continue
    report = json.loads((directory/'report.json').read_text())
    if report['name']=='original_cats':
        report['seconds'].pop('generation',None)
        report['generation_note']='Replayed actual generated IDs from https://github.com/frgfm/torch-cam/actions/runs/37996221308; no new generation was timed.'
    if report['finite_differences']:
        report['finite_difference_target'] = dict(position=1,id=report['explained_ids'][0],text=report['token_text'][0],
            note='First generated token; checks partial attention derivatives, not noun faithfulness.')
    report['tam_context_convention']='Earlier generated IDs are paired with the final states that predicted them, following TorchCAM docs and the official TAM implementation.'
    (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    arrays = np.load(directory/'maps.npz',allow_pickle=False)
    image = Image.open(directory/'input.png').convert('RGB')
    selected = list(report['selected'].items())
    if not selected:
        continue
    fig,axes = plt.subplots(len(selected),4,figsize=(14,3.3*len(selected)),squeeze=False,layout='constrained')
    for row,(target,index) in enumerate(selected):
        panels = [('input','Input image'),('dexar','DEX-AR'),('tam1','TorchCAM TAM · no smoothing'),('tam3','TorchCAM TAM · default filter')]
        for axis,(key,title) in zip(axes[row],panels):
            axis.imshow(image)
            if key!='input':
                heatmap = torch.from_numpy(arrays[key][0,index])
                resized = F.interpolate(heatmap[None,None],image.size[::-1],mode='bilinear',align_corners=False)[0,0]
                axis.imshow(resized.numpy(),cmap='turbo',vmin=0,vmax=1,alpha=.45)
            axis.set_title(title,fontsize=10)
            axis.axis('off')
        axes[row,0].text(.01,.03,f"Token {index+1}: {report['token_text'][index]!r}\nID {report['explained_ids'][index]}",
                            transform=axes[row,0].transAxes,color='white',fontsize=10,
                            bbox=dict(facecolor='black',alpha=.7,edgecolor='none'))
        word_probes = [p for p in report['probes'] if p['target']==target]
        other = arrays['banana_at_'+str(index)][0]
        actual = arrays['dexar'][0,index]
        correlation = float(np.corrcoef(actual.ravel(),other.ravel())[0,1])
        summary.append(dict(case=report['name'],target=target,position=index+1,token_id=report['explained_ids'][index],
                            target_control_correlation=correlation,
                            **{p['method']+'_logit_drop':p['logit_drop'] for p in word_probes},
                            **{p['method']+'_delta_nll':p['delta_nll'] for p in word_probes}))
    fig.suptitle('Actual answer: '+textwrap.fill(report['answer'],105)+'\nPrompt: '+textwrap.fill(report['prompt'],110),fontsize=11)
    fig.savefig(directory/'comparison.png',dpi=140,facecolor='white')
    plt.close(fig)
    fig,axes = plt.subplots(len(selected),2,figsize=(8,3.2*len(selected)),squeeze=False,layout='constrained')
    for row,(target,index) in enumerate(selected):
        for axis,heatmap,title in zip(axes[row],[arrays['dexar'][0,index],arrays['banana_at_'+str(index)][0]],
                [f"Actual {report['token_text'][index]!r}, ID {report['explained_ids'][index]}","Control: ' banana' · same prefix · not generated"]):
            resized = F.interpolate(torch.from_numpy(heatmap)[None,None],image.size[::-1],mode='bilinear',align_corners=False)[0,0]
            axis.imshow(image);axis.imshow(resized.numpy(),cmap='turbo',vmin=0,vmax=1,alpha=.45)
            axis.set_title(title,fontsize=10);axis.axis('off')
    fig.suptitle('DEX-AR target specificity check · identical image and predicting state',fontsize=11)
    fig.savefig(directory/'target-control.png',dpi=140,facecolor='white')
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(8,3.3),layout='constrained')
    for axis,title in zip(axes,['Input image','DEX-AR answer map']):
        axis.imshow(image);axis.set_title(title);axis.axis('off')
    resized = F.interpolate(torch.from_numpy(arrays['sequence'])[0,None,None],image.size[::-1],mode='bilinear',align_corners=False)[0,0]
    axes[1].imshow(resized.numpy(),cmap='turbo',vmin=0,vmax=1,alpha=.45)
    fig.suptitle(textwrap.fill(report['answer'],90),fontsize=11)
    fig.savefig(directory/'sequence.png',dpi=140,facecolor='white');plt.close(fig)
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
labels=[s['case'].replace('_caption','').replace('catdog_','cat+dog / ').replace('original_cats','original')+' / '+s['target'] for s in summary]
fig,axes=plt.subplots(1,2,figsize=(13,7),layout='constrained',sharey=True)
positions=np.arange(len(summary))
for axis,field,title in zip(axes,['logit_drop','delta_nll'],['Raw target logit drop','Conditional negative log-probability increase']):
    for offset,method,label,color in [(-.24,'dexar','DEX-AR','#e76f51'),(0,'tam3','TorchCAM TAM (default)','#2a9d8f'),(.24,'random','Random cells (seed 0)','#999999')]:
        axis.barh(positions+offset,[s[method+'_'+field] for s in summary],height=.23,label=label,color=color)
    axis.axvline(0,color='black',linewidth=.7)
    axis.set_title(title,fontsize=11)
    axis.grid(axis='x',alpha=.2)
    axis.set_axisbelow(True)
axes[0].set_yticks(positions,labels,fontsize=9);axes[0].invert_yaxis();axes[1].legend(fontsize=9)
fig.suptitle('Mask top 20% of native cells with mean image colour · freeze the actual preceding answer tokens\nPositive: target score/confidence fell. Negative: it rose. Demonstrations, not an accuracy benchmark.',fontsize=11)
fig.savefig(ROOT/'masking-controls.png',dpi=140,facecolor='white');plt.close(fig)
from PIL import ImageDraw
annotation_path=ROOT/'coco_annotations.json'
if annotation_path.exists():
    directory=ROOT/'original_cats'
    image=Image.open(directory/'input.png')
    arrays=np.load(directory/'maps.npz',allow_pickle=False)
    report=json.loads((directory/'report.json').read_text())
    annotations=json.loads(annotation_path.read_text())
    energy=[]
    for word,category in [('cats',17),('couch',63)]:
        mask=Image.new('L',image.size,0)
        draw=ImageDraw.Draw(mask)
        for annotation in annotations:
            if annotation['image_id']==39769 and annotation['category_id']==category:
                for polygon in annotation['segmentation']:
                    draw.polygon(list(zip(polygon[::2],polygon[1::2])),fill=1)
        truth=np.asarray(mask).astype(bool)
        index=report['selected'][word]
        for key,label in [('dexar','DEX-AR'),('tam1','TAM without smoothing'),('tam3','TAM default')]:
            resized=F.interpolate(torch.from_numpy(arrays[key][0,index])[None,None],image.size[::-1],mode='bilinear',align_corners=False)[0,0].numpy()
            energy.append(dict(target=word,method=label,energy_in_mask=float(resized[truth].sum()/resized.sum()),object_area_fraction=float(truth.mean())))
    (ROOT/'original-localization.json').write_text(json.dumps(dict(source='https://raw.githubusercontent.com/huggingface/transformers/main/tests/fixtures/tests_samples/COCO/coco_annotations.txt',
        method='Union of official COCO polygons rasterized at original resolution. Bilinear heatmap energy within the mask. One-image diagnostic, not IoU or an accuracy benchmark.',results=energy),indent=2)+'\n')
print(json.dumps(dict(cases=len(set(s['case'] for s in summary)),targets=len(summary))))

import gc
import json
import sys
import time
import traceback
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path('demo').resolve()))
import inference
from torchcam.methods import GradCAM, LayerCAM, TokenIG

out = Path('demo-diagnostics')
out.mkdir(exist_ok=True)
report = {'torch': torch.__version__, 'initial_threads': torch.get_num_threads()}
torch.set_num_threads(2)

def save():
    (out / 'report.json').write_text(json.dumps(report, indent=2))

def event(**kwargs):
    print(json.dumps(kwargs), flush=True)

def decision(name):
    event(phase='load_decision', model=name)
    started = time.perf_counter()
    resource, tokenizer, lock = inference.load_resource('decision', name, 'cpu')
    elapsed = time.perf_counter() - started
    model = resource.model if name == 'laya' else resource
    entry = {'load_seconds': elapsed, 'parameters': sum(p.numel() for p in model.parameters())}
    report[name] = entry
    count = [0]
    started = time.perf_counter()
    def budget(*_):
        count[0] += 1
        if time.perf_counter() - started > 90:
            raise TimeoutError('Attribution exceeded diagnostic 90-second budget')
    hook = model.register_forward_pre_hook(budget)
    try:
        result = inference.explain_decision('I was charged twice for the same order. Please refund the duplicate payment.', name, 'cpu')
        entry.update(elapsed=time.perf_counter()-started, forward_calls=count[0], result=result)
    except Exception as error:
        entry.update(elapsed=time.perf_counter()-started, forward_calls=count[0], error=repr(error))
    finally:
        hook.remove()
    event(model=name, load_seconds=elapsed, seconds=entry['elapsed'], forwards=count[0], error=entry.get('error'), steps=entry.get('result',{}).get('steps'))
    save()
    del model, resource
    inference.load_resource.clear()
    gc.collect()

def vlm():
    image_url='https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg'
    with urlopen(Request(image_url, headers={'User-Agent':'TorchCAM demo validation'}),timeout=60) as response:
        (out/'border-collie.jpg').write_bytes(response.read())
    image=Image.open(out/'border-collie.jpg').convert('RGB')
    event(phase='load_vlm', original_image_size=image.size)
    started=time.perf_counter()
    model,processor,lock=inference.load_resource('vlm',inference.VLM_ID,'cpu')
    entry={'load_seconds':time.perf_counter()-started,'image_size':image.size,'config':{'vision':model.config.vision_config.to_dict(),'scale_factor':model.config.scale_factor}}
    report['vlm']=entry
    started=time.perf_counter()
    result=inference.generate_answer(image,'Describe this image in one sentence.','cpu')
    entry.update(answer=result['answer'],generation_seconds=time.perf_counter()-started)
    result['input_image'].save(out/'model-input.png')
    side=result['grid_side']
    factor=model.config.scale_factor
    n=side*factor
    raster=torch.arange(n*n,dtype=torch.float32).reshape(1,n*n,1)
    actual=model.model.connector.pixel_shuffle(raster,factor).mean(-1)
    expected=torch.nn.functional.avg_pool2d(raster.reshape(1,1,n,n),factor).reshape(1,-1)
    torch.testing.assert_close(actual,expected)
    entry['spatial_order_verified']=True
    decoded={i:processor.tokenizer.decode([result['generated_ids'][i]]) for i in result['choices']}
    selected=[i for i,s in decoded.items() if any(word in s.lower() for word in ['dog','black','white','sitting','grass'])][:4]
    if not selected:selected=result['choices'][:2]
    entry['tokens']={}
    for index in selected:
        token_id=result['generated_ids'][index]
        token=decoded[index]
        maps={}
        record={}
        started=time.perf_counter()
        maps['TAM'],probability=inference.explain_answer_token(result,index,'cpu')
        record['TAM_seconds']=time.perf_counter()-started
        inputs={key:value.clone() for key,value in result['inputs'].items()}
        prefix=torch.tensor([result['generated_ids'][:index]],dtype=torch.long)
        inputs['input_ids']=torch.cat((inputs['input_ids'],prefix),-1)
        inputs['attention_mask']=torch.cat((inputs['attention_mask'],torch.ones_like(prefix)),-1)
        for name,method in [('GradCAM',GradCAM),('LayerCAM',LayerCAM)]:
            current={key:value.clone() for key,value in inputs.items()}
            current['pixel_values'].requires_grad_(True)
            transform=lambda states:states.transpose(1,2).reshape(states.shape[0],states.shape[-1],side,side)
            started=time.perf_counter()
            with method(model,'model.connector',reshape_transform=transform) as extractor:
                output=model(**current,use_cache=False,logits_to_keep=1)
                maps[name]=extractor(token_id,output.logits[:,-1,:])[0].detach()
            record[name+'_seconds']=time.perf_counter()-started
        with torch.inference_mode():
            original=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1,token_id].item()
            drops={}
            for name,cam in maps.items():
                values=cam.flatten()
                k=max(1,int(values.numel()*.2))
                rows={}
                for ranking in ['top','bottom']:
                    indices=values.topk(k,largest=ranking=='top').indices
                    mask=torch.zeros(side*side,dtype=torch.bool)
                    mask[indices]=True
                    h,w=inputs['pixel_values'].shape[-2:]
                    mask=mask.reshape(side,side).repeat_interleave(h//side,0).repeat_interleave(w//side,1)
                    altered={key:value.clone() for key,value in inputs.items()}
                    altered['pixel_values'][...,mask]=0
                    score=model(**altered,use_cache=False,logits_to_keep=1).logits[0,-1,token_id].item()
                    rows[ranking]=original-score
                drops[name]=rows
            record.update(probability=probability,logit=original,occlusion_logit_drop=drops)
        entry['tokens'][f'{index}:{token}']=record
        fig,axes=plt.subplots(1,4,figsize=(12,3))
        axes[0].imshow(result['input_image']);axes[0].set_title('Model input')
        for ax,(name,cam) in zip(axes[1:],maps.items()):
            ax.imshow(result['input_image'])
            ax.imshow(cam.squeeze().numpy(),cmap='jet',alpha=.55,extent=(0,result['input_image'].width,result['input_image'].height,0))
            ax.set_title(name)
        for ax in axes:ax.axis('off')
        fig.suptitle(f'{token!r} · p={probability:.3f}')
        fig.tight_layout();fig.savefig(out/f'token-{index}.png',dpi=150);plt.close(fig)
        event(token=token,index=index,**record)
        save()
    event(answer=result['answer'],generation_seconds=entry['generation_seconds'],grid=side)
    save()

for name,operation in [('vlm',vlm),('minilm',lambda:decision('minilm')),('laya',lambda:decision('laya'))]:
    try:operation()
    except Exception as error:
        report.setdefault(name,{})['error']=repr(error)
        traceback.print_exc()
        save()
    inference.load_resource.clear()
    gc.collect()
save()

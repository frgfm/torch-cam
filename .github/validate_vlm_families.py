import json, os, sys, time
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps
sys.path.insert(0,str(Path('demo').resolve()))
import inference
from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask

torch.set_num_threads(2)
out=Path('family-validation');out.mkdir(exist_ok=True)
model_id=os.environ['MODEL_ID'];report={'model':model_id,'images':{}}
def save(): (out/'report.json').write_text(json.dumps(report,indent=2))
model,processor,_=inference.load_resource('vlm',model_id,'cpu')
original_generate=model.generate
original_chat=processor.apply_chat_template
family=model.config.model_type

def chat(*args,**kwargs):
 kwargs.pop('max_pixels',None);kwargs.pop('max_soft_tokens',None)
 if family=='qwen3_vl':kwargs.pop('enable_thinking',None)
 kwargs['processor_kwargs']={'max_pixels':384**2} if family=='qwen3_vl' else {'max_soft_tokens':140}
 return original_chat(*args,**kwargs)

images=[('collie',Image.open('fixtures/border-collie.jpg').convert('RGB'))]
from io import BytesIO
images.append(('cats',Image.open(BytesIO(urlopen('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png',timeout=60).read())).convert('RGB')))
for image_name,image in images:
 with patch.object(model,'generate',side_effect=lambda **kw:original_generate(**{**kw,'max_new_tokens':48})),patch.object(processor,'apply_chat_template',side_effect=chat):
  result=inference.generate_answer(image,'a picture of','cpu',model_id)
 result['input_image'].save(out/f'{image_name}-input.png')
 words=[(i,processor.tokenizer.decode([result['generated_ids'][i]])) for i in result['choices']]
 image_report={'caption':result['answer'],'grid':result['grid_shape'],'words':words,'tokens':{}}
 report['images'][image_name]=image_report;save();print(image_name,json.dumps(image_report),flush=True)
 indices=[i for i,word in words if any(noun in word.strip().lower() for noun in ['dog','border','cat'])][:1]
 indices += [i for i,word in words if word.strip().lower()=='black'][:1]
 assert indices,words
 for index in dict.fromkeys(indices):
  token_id=result['generated_ids'][index];word=processor.tokenizer.decode([token_id]);data=inference.answer_inputs(result,index,'cpu')
  layer,reshape,order,grid=inference.vlm_target(model,data)
  if family=='qwen3_vl':
   visual=model.model.visual
   layers=[visual.patch_embed,visual.blocks[min(model.config.vision_config.deepstack_visual_indexes)],visual.blocks[-1]]
   names=['patch','first-deepstack','last-block']
  else:
   visual=model.model.vision_tower
   layers=[visual.patch_embedder.input_proj,visual.patch_embedder,visual.encoder.layers[-1]]
   names=['patch-projection','patch-position','last-block']
  maps={}
  with LayerCAM(model,layers,reshape_transform=reshape) as extractor:
   logits=model(**data,use_cache=False,logits_to_keep=1).logits[:,-1].float()
   baseline={'logit':float(logits[0,token_id].detach()),'logprob':float(logits[0].log_softmax(-1)[token_id].detach())}
   for objective in ['logit','logprob','logodds']:
    if objective=='logodds':
     target=lambda sample:sample[token_id]-torch.logsumexp(torch.cat([sample[:token_id],sample[token_id+1:]]),0)
     cams=extractor(scores=logits,targets=target)
    else:
     scores=logits if objective=='logit' else logits.log_softmax(-1)
     cams=extractor(token_id,scores,retain_graph=True)
    for name,cam in zip(names,cams):maps[objective+'-'+name]=cam.detach()
  token_report={'baseline':baseline,'maps':{}};image_report['tokens'][word]=token_report
  data['pixel_values']=data['pixel_values'].detach()
  with torch.no_grad():
   for name,cam in maps.items():
    row={'max':float(cam.max()),'occlusion':{}}
    overlay_mask(result['input_image'],to_pil_image(cam,mode='F'),alpha=.5).save(out/f'{image_name}-{index}-{name}.png')
    for largest,label in [(True,'top'),(False,'bottom')]:
     positions=order[cam.flatten().topk(max(1,cam.numel()//5),largest=largest).indices]
     changed={**data,'pixel_values':data['pixel_values'].clone()}
     flat=changed['pixel_values'].reshape(-1,changed['pixel_values'].shape[-1]);flat[positions]=0 if family=='qwen3_vl' else .5
     scores=model(**changed,use_cache=False,logits_to_keep=1).logits[0,-1].float()
     row['occlusion'][label]={'logit_drop':baseline['logit']-float(scores[token_id]),'logprob_drop':baseline['logprob']-float(scores.log_softmax(-1)[token_id])}
    token_report['maps'][name]=row;save();print(image_name,word,name,json.dumps(row),flush=True)

import json,os,sys,time
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen
from io import BytesIO
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
sys.path.insert(0,str(Path('demo').resolve()))
import inference
from torchcam.methods import GradCAM,LayerCAM,TAM
from torchcam.utils import overlay_mask

torch.set_num_threads(2)
out=Path('family-validation');out.mkdir(exist_ok=True)
model_id=os.environ['MODEL_ID'];report={'model':model_id,'images':{}}
def save():(out/'report.json').write_text(json.dumps(report,indent=2))
model,processor,_=inference.load_resource('vlm',model_id,'cpu')
family=model.config.model_type
original_generate=model.generate;original_chat=processor.apply_chat_template
def chat(*args,**kwargs):
 kwargs.pop('max_pixels',None);kwargs.pop('max_soft_tokens',None)
 if family=='qwen3_vl':kwargs.pop('enable_thinking',None)
 kwargs['processor_kwargs']={'max_pixels':384**2} if family=='qwen3_vl' else {'max_soft_tokens':140}
 return original_chat(*args,**kwargs)
images=[('collie',Image.open('fixtures/border-collie.jpg').convert('RGB')),('cats',Image.open(BytesIO(urlopen('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png',timeout=60).read())).convert('RGB'))]
for name,image in images:
 with patch.object(model,'generate',side_effect=lambda **kw:original_generate(**{**kw,'max_new_tokens':48})),patch.object(processor,'apply_chat_template',side_effect=chat):result=inference.generate_answer(image,'a picture of','cpu',model_id)
 result['input_image'].save(out/f'{name}-input.png')
 words=[(i,processor.tokenizer.decode([result['generated_ids'][i]])) for i in result['choices']]
 row={'caption':result['answer'],'words':words,'tokens':{}};report['images'][name]=row;save()
 indices=[i for i,word in words if any(n in word.lower() for n in ['dog','border','cat'])][:1]
 for index in indices:
  word=processor.tokenizer.decode([result['generated_ids'][index]]);token_id=result['generated_ids'][index]
  data=inference.answer_inputs(result,index,'cpu');_,_,order,(height,width)=inference.vlm_target(model,data)
  if family=='qwen3_vl':
   visual=model.model.visual;layers=[visual.merger,*visual.deepstack_merger_list];pool=model.config.vision_config.spatial_merge_size
  else:layers=[model.model.embed_vision];pool=model.config.vision_config.pooling_kernel_size
  h,w=height//pool,width//pool
  reshape=lambda states:states.float().reshape(1,h,w,-1).permute(0,3,1,2)
  maps={};target=lambda sample:sample[token_id]-torch.logsumexp(torch.cat([sample[:token_id],sample[token_id+1:]]),0)
  with GradCAM(model,layers,reshape_transform=reshape) as grad,LayerCAM(model,layers,reshape_transform=reshape) as layer:
   output=model(**data,use_cache=False,logits_to_keep=1,output_hidden_states=True)
   logits=output.logits[:,-1].float()
   maps['GradCAM-projected']=grad.fuse_cams(grad(scores=logits,targets=target,retain_graph=True))
   maps['LayerCAM-projected']=layer.fuse_cams(layer(scores=logits,targets=target))
  ids=data['input_ids'][0];visual_mask=ids==model.config.image_token_id
  special=torch.tensor(processor.tokenizer.all_special_ids,device=ids.device)
  context_mask=~visual_mask & ~torch.isin(ids,special)
  states=output.hidden_states[-1].detach()
  maps['TAM']=TAM(model.get_output_embeddings())(token_id,states[:,visual_mask].reshape(1,h,w,-1),states[:,context_mask],data['input_ids'][:,context_mask])
  token_row={'probability':float(logits[0].softmax(-1)[token_id].detach()),'maps':{}};row['tokens'][word]=token_row
  baseline=float(logits[0].log_softmax(-1)[token_id].detach());data['pixel_values']=data['pixel_values'].detach()
  with torch.no_grad():
   for method,cam in maps.items():
    current={};overlay_mask(result['input_image'],to_pil_image(cam,mode='F'),alpha=.5).save(out/f'{name}-{index}-{method}.png')
    for biggest,label in [(True,'top'),(False,'bottom')]:
     mask=torch.zeros(h*w,dtype=torch.bool);mask[cam.flatten().topk(max(1,cam.numel()//5),largest=biggest).indices]=True
     mask=mask.reshape(h,w).repeat_interleave(pool,0).repeat_interleave(pool,1).flatten()
     modified={**data,'pixel_values':data['pixel_values'].clone()};flat=modified['pixel_values'].reshape(-1,modified['pixel_values'].shape[-1]);flat[order[mask]]=0 if family=='qwen3_vl' else .5
     scores=model(**modified,use_cache=False,logits_to_keep=1).logits[0,-1].float()
     current[label]=baseline-float(scores.log_softmax(-1)[token_id])
    token_row['maps'][method]=current;save();print(name,word,method,json.dumps(current),flush=True)

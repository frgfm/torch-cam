import json,sys,time
from contextlib import ExitStack
from pathlib import Path
from urllib.request import urlopen
from io import BytesIO
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
sys.path.insert(0,str(Path('demo').resolve()))
import inference
from torchcam.methods import GradCAM,LayerCAM,XGradCAM
from torchcam.utils import overlay_mask
torch.set_num_threads(2)
out=Path('family-validation');out.mkdir(exist_ok=True)
report={'bf16':getattr(torch.cpu,'_is_avx512_bf16_supported',lambda:False)(),'images':{}}
def save():(out/'report.json').write_text(json.dumps(report,indent=2))
print(report,flush=True);save()
if not report['bf16']:sys.exit()
mid='google/gemma-4-E2B-it';model,processor,_=inference.load_resource('vlm',mid,'cpu')
images=[('collie',Image.open('fixtures/border-collie.jpg').convert('RGB')),('cats',Image.open(BytesIO(urlopen('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png',timeout=60).read())).convert('RGB'))]
for name,image in images:
 result=inference.generate_answer(image,'Describe the image in one short sentence.','cpu',mid)
 words=[(i,processor.tokenizer.decode([result['generated_ids'][i]])) for i in result['choices']]
 index=next(i for i,word in words if word.strip().lower() in ('dog','border','cat','cats'));token=result['generated_ids'][index]
 row={'caption':result['answer'],'word':processor.tokenizer.decode([token]),'maps':{}};report['images'][name]=row;save();print(name,row['caption'],flush=True)
 data=inference.answer_inputs(result,index,'cpu');data['pixel_values']=data['pixel_values'].detach()
 order,(height,width)=inference.vlm_layout(model,data);pool=model.config.vision_config.pooling_kernel_size
 candidates=[('projected',model.model.embed_vision,True),('last-vision',model.model.vision_tower.encoder.layers[-1],False),('last-norm',model.model.vision_tower.encoder.layers[-1].input_layernorm,False)]
 for label,layer,pooled in candidates:
  h,w=(height//pool,width//pool) if pooled else (height,width)
  def reshape(states):
   if not pooled:states=states.reshape(-1,states.shape[-1])[order]
   return states.float().reshape(1,h,w,-1).permute(0,3,1,2)
  target=lambda sample:sample[token]-torch.logsumexp(torch.cat([sample[:token],sample[token+1:]]),0)
  maps={}
  with ExitStack() as stack:
   stack.enter_context(layer.register_forward_hook(lambda _,__,output:output.detach().requires_grad_(True)))
   extractors=[stack.enter_context(method(model,layer,reshape_transform=reshape)) for method in (GradCAM,LayerCAM,XGradCAM)]
   output=model(**data,use_cache=False,logits_to_keep=1);logits=output.logits[:,-1].float()
   baseline=float(logits[0].log_softmax(-1)[token].detach())
   for k,extractor in enumerate(extractors):maps[type(extractor).__name__]=extractor(scores=logits,targets=target,retain_graph=k<2)[0].detach()
  for method,cam in maps.items():
   key=f'{label}-{method}';current={};row['maps'][key]=current
   overlay_mask(result['input_image'],to_pil_image(cam,mode='F'),alpha=.5).save(out/f'{name}-{key}.png')
   with torch.no_grad():
    for fraction in (.2,.5):
     pair={}
     for biggest,which in [(True,'top'),(False,'bottom')]:
      mask=torch.zeros(h*w,dtype=torch.bool);mask[cam.flatten().topk(max(1,int(cam.numel()*fraction)),largest=biggest).indices]=True
      if pooled:mask=mask.reshape(h,w).repeat_interleave(pool,0).repeat_interleave(pool,1).flatten()
      modified={**data,'pixel_values':data['pixel_values'].clone()};flat=modified['pixel_values'].reshape(-1,modified['pixel_values'].shape[-1]);flat[order[mask]]=.5
      after=model(**modified,use_cache=False,logits_to_keep=1).logits[0,-1].float().log_softmax(-1)[token]
      pair[which]=baseline-float(after)
     current[str(fraction)]=pair
   current['rank']=sum(p['top']-max(0,p['bottom']) for f,p in current.items() if f in ['0.2','0.5'])/2
   print(name,key,json.dumps(current),flush=True);save()
assert not any(m._forward_hooks for m in model.modules())

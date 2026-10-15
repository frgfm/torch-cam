import gc,json,time
from pathlib import Path
from urllib.request import Request,urlopen
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
from transformers import AutoProcessor,BlipForConditionalGeneration
from torchcam.methods import GradCAM,LayerCAM

torch.set_num_threads(2)
out=Path('demo-diagnostics');out.mkdir(exist_ok=True)
model_id='Salesforce/blip-image-captioning-base'
started=time.perf_counter()
processor=AutoProcessor.from_pretrained(model_id)
model=BlipForConditionalGeneration.from_pretrained(model_id).eval().requires_grad_(False)
report={'model':model_id,'load_seconds':time.perf_counter()-started,'images':{}}
for label,url in [('collie','https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg'),('cats','https://images.cocodataset.org/val2017/000000039769.jpg')]:
 with urlopen(Request(url,headers={'User-Agent':'TorchCAM demo validation'}),timeout=60) as response:(out/(label+'.jpg')).write_bytes(response.read())
 original=Image.open(out/(label+'.jpg')).convert('RGB')
 size=model.config.vision_config.image_size
 image=ImageOps.fit(original,(size,size))
 inputs=processor(images=image,text='a picture of',return_tensors='pt')
 start=inputs['input_ids'].shape[-1]-1
 started=time.perf_counter()
 with torch.inference_mode():sequences=model.generate(**inputs,do_sample=False,max_new_tokens=32)
 caption=processor.decode(sequences[0],skip_special_tokens=True)
 row={'caption':caption,'generation_seconds':time.perf_counter()-started,'tokens':{}}
 report['images'][label]=row
 indices=[i for i in range(start,sequences.shape[1]) if any(w in processor.decode([sequences[0,i]]).lower() for w in ['dog','cat','sitting','black','white'])][:4]
 grid=size//model.config.vision_config.patch_size
 for index in indices:
  token_id=int(sequences[0,index]);token=processor.decode([token_id])
  current={'pixel_values':inputs['pixel_values'].clone(),'input_ids':sequences[:,:index].clone()}
  current['attention_mask']=torch.ones_like(current['input_ids'])
  maps={};timings={}
  for name,kind,target,transform in [
    ('VisionGradCAM',GradCAM,'vision_model.post_layernorm',lambda x:x[:,1:].transpose(1,2).reshape(x.shape[0],x.shape[-1],grid,grid)),
    ('Cross6LayerCAM',LayerCAM,'text_decoder.bert.encoder.layer.6.crossattention.self.dropout',lambda x:x[:,:,-1,1:].reshape(x.shape[0],x.shape[1],grid,grid)),
    ('Cross11LayerCAM',LayerCAM,'text_decoder.bert.encoder.layer.11.crossattention.self.dropout',lambda x:x[:,:,-1,1:].reshape(x.shape[0],x.shape[1],grid,grid)),
  ]:
   data={key:value.clone() for key,value in current.items()};data['pixel_values'].requires_grad_(True)
   started=time.perf_counter()
   with kind(model,target,reshape_transform=transform) as extractor:
    output=model(**data,logits_to_keep=1,use_cache=False)
    maps[name]=extractor(token_id,output.logits[:,-1])[0].detach()
   timings[name]=time.perf_counter()-started
  with torch.inference_mode():
   output=model(**current,logits_to_keep=1,use_cache=False)
   logit=float(output.logits[0,-1,token_id]);probability=float(output.logits[0,-1].softmax(-1)[token_id])
   drops={}
   for name,cam in maps.items():
    values=cam.flatten();k=max(1,int(values.numel()*.2));pair={}
    for mode in ['top','bottom']:
     positions=values.topk(k,largest=mode=='top').indices
     mask=torch.zeros(grid*grid,dtype=torch.bool);mask[positions]=True
     mask=mask.reshape(grid,grid).repeat_interleave(size//grid,0).repeat_interleave(size//grid,1)
     altered={key:value.clone() for key,value in current.items()};altered['pixel_values'][...,mask]=0
     pair[mode]=logit-float(model(**altered,logits_to_keep=1,use_cache=False).logits[0,-1,token_id])
    drops[name]=pair
  entry={'probability':probability,'seconds':timings,'occlusion_logit_drop':drops};row['tokens'][f'{index}:{token}']=entry
  print(json.dumps({'image':label,'token':token,**entry}),flush=True)
  fig,axes=plt.subplots(1,4,figsize=(12,3));axes[0].imshow(image);axes[0].set_title('Model input')
  for ax,(name,cam) in zip(axes[1:],maps.items()):
   ax.imshow(image);ax.imshow(cam.squeeze().numpy(),cmap='jet',alpha=.55,extent=(0,size,size,0));ax.set_title(name)
  for ax in axes:ax.axis('off')
  fig.suptitle(f'{label}: {token!r} · p={probability:.3f}');fig.tight_layout();fig.savefig(out/f'{label}-{index}.png',dpi=150);plt.close(fig)
  (out/'report.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({'image':label,'caption':caption,'generation_seconds':row['generation_seconds']}),flush=True)
(out/'report.json').write_text(json.dumps(report,indent=2))

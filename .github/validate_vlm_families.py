import gc,json,os,sys,time
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask
sys.path.insert(0,str(Path('demo').resolve()))
import inference

torch.set_num_threads(2)
out=Path('family-validation');out.mkdir(exist_ok=True)
model_id=os.environ['MODEL_ID'];report={'model':model_id,'cpu':torch.backends.cpu.get_cpu_capability(),'bf16':getattr(torch.cpu,'_is_avx512_bf16_supported',lambda:False)(),'images':{}}
def save():(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True);save()
if 'gemma' in model_id and not report['bf16']:
 report['skipped']='CPU has no BF16 acceleration';save();sys.exit()
model,processor,_=inference.load_resource('vlm',model_id,'cpu')
images=[('collie',Image.open('fixtures/border-collie.jpg').convert('RGB')),('cats',Image.open(BytesIO(urlopen('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png',timeout=60).read())).convert('RGB'))]
for name,image in images:
 started=time.perf_counter();result=inference.generate_answer(image,'a picture of','cpu',model_id)
 row={'caption':result['answer'],'generation_seconds':time.perf_counter()-started,'tokens':{}};report['images'][name]=row;save();print(name,result['answer'],flush=True)
 result['input_image'].save(out/f'{name}-input.png')
 indices=[i for i in result['choices'] if any(word in processor.tokenizer.decode([result['generated_ids'][i]]).strip().lower() for word in ['dog','border','cat'])][:1]
 assert indices,'No subject token'
 for index in indices:
  word=processor.tokenizer.decode([result['generated_ids'][index]]);token=result['generated_ids'][index]
  started=time.perf_counter();cam,prob=inference.explain_answer_token(result,index,'cpu')
  current={'seconds':time.perf_counter()-started,'probability':prob,'occlusion':{}}
  row['tokens'][word]=current;save()
  assert cam.isfinite().all() and cam.any()
  overlay_mask(result['input_image'],to_pil_image(cam,mode='F'),alpha=.5).save(out/f'{name}-overlay.png')
  data=inference.answer_inputs(result,index,'cpu');data['pixel_values']=data['pixel_values'].detach()
  _,_,groups,_=inference.vlm_target(model,data)
  with torch.no_grad():
   before=model(**data,use_cache=False,logits_to_keep=1).logits[0,-1].float().log_softmax(-1)[token]
   assert abs(float(before.exp())-prob)<1e-6
   for fraction in [.2,.5]:
    pair={}
    for largest,label in [(True,'top'),(False,'bottom')]:
     selected=cam.flatten().topk(max(1,int(cam.numel()*fraction)),largest=largest).indices
     modified={**data,'pixel_values':data['pixel_values'].clone()};flat=modified['pixel_values'].reshape(-1,modified['pixel_values'].shape[-1]);flat[groups[selected].flatten()]=0 if model.config.model_type=='qwen3_vl' else .5
     after=model(**modified,use_cache=False,logits_to_keep=1).logits[0,-1].float().log_softmax(-1)[token]
     pair[label]=float(before-after)
    current['occlusion'][str(fraction)]=pair;save();print(name,word,fraction,json.dumps(pair),flush=True)
  current['top_mean']=sum(pair['top'] for pair in current['occlusion'].values())/2
  current['bottom_mean']=sum(pair['bottom'] for pair in current['occlusion'].values())/2
  current['rank_check']=current['top_mean']>max(0,current['bottom_mean'])
  save()
assert not any(m._forward_hooks for m in model.modules()),'Leaked hooks'

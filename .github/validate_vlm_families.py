import json,sys
from pathlib import Path
import torch
from PIL import Image
sys.path.insert(0,str(Path('demo').resolve()))
import inference
torch.set_num_threads(2)
result=inference.generate_answer(Image.open('fixtures/border-collie.jpg').convert('RGB'),'a picture of','cpu')
model,processor,_=inference.load_resource('vlm',inference.VLM_ID,'cpu')
i=next(i for i in result['choices'] if processor.tokenizer.decode([result['generated_ids'][i]]).strip().lower()=='dog')
cam,p=inference.explain_answer_token(result,i,'cpu');inputs=inference.answer_inputs(result,i,'cpu');inputs['pixel_values']=inputs['pixel_values'].detach();token=result['generated_ids'][i]
report={'caption':result['answer'],'probability':p,'occlusion':{}}
with torch.no_grad():
 before=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float().log_softmax(-1)[token]
 assert abs(float(before.exp())-p)<1e-6
 for fraction in (.2,.5):
  pair={}
  for largest,label in [(True,'top'),(False,'bottom')]:
   selected=cam.flatten().topk(max(1,int(cam.numel()*fraction)),largest=largest).indices
   h,w=result['grid_shape'];mask=torch.zeros(h*w,dtype=torch.bool);mask[selected]=True
   mask=mask.reshape(h,w).repeat_interleave(result['input_image'].height//h,0).repeat_interleave(result['input_image'].width//w,1)
   altered={**inputs,'pixel_values':inputs['pixel_values'].clone()};altered['pixel_values'][...,mask]=0
   after=model(**altered,use_cache=False,logits_to_keep=1).logits[0,-1].float().log_softmax(-1)[token]
   pair[label]=float(before-after)
  report['occlusion'][str(fraction)]=pair
print(json.dumps(report),flush=True)
assert sum(p['top'] for p in report['occlusion'].values())>max(0,sum(p['bottom'] for p in report['occlusion'].values()))
assert not any(m._forward_hooks for m in model.modules())

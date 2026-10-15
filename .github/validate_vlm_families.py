import gc, json, logging, os, sys, time
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path('demo').resolve()))
import inference
from PIL import Image
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps

torch.set_num_threads(2)
logging.basicConfig(level=logging.INFO)
model_id = os.environ['MODEL_ID']
out = Path('family-validation'); out.mkdir(exist_ok=True)
report = {'model':model_id}
def save():
 (out/'report.json').write_text(json.dumps(report, indent=2))
try:
 started=time.perf_counter()
 if model_id in ('deberta','decider'):
  for text, expected in [('I was charged twice. Please refund the duplicate payment.','Billing'), ('My password reset link has expired. Help me sign in.','Account access'), ('My shipment has not arrived. The tracking page shows a delivery delay.','Delivery')]:
   result=inference.explain_decision(text, model_id, 'cpu')
   predicted=result['labels'][max(range(3), key=result['probabilities'].__getitem__)]
   assert predicted==expected, (predicted,expected)
   report[expected] = result
  report['total_seconds']=time.perf_counter()-started
  save(); print(json.dumps(report),flush=True);sys.exit()
 image=Image.open('fixtures/border-collie.jpg').convert('RGB')
 model, processor, _ = inference.load_resource('vlm',model_id,'cpu')
 report['load_seconds']=time.perf_counter()-started;save()
 started=time.perf_counter()
 result=inference.generate_answer(image,'a picture of','cpu',model_id)
 report.update(caption=result['answer'],generation_seconds=time.perf_counter()-started,grid=result['grid_shape'],tokens={})
 result['input_image'].save(out/'model-input.png');save()
 print(json.dumps(report),flush=True)
 indices=[i for i in result['choices'] if any(w in processor.tokenizer.decode([result['generated_ids'][i]]).lower() for w in ['dog','black','white','sitting'])][:3]
 assert indices, 'No expected image-sensitive words in caption'
 for index in indices:
  token_id=result['generated_ids'][index];token=processor.tokenizer.decode([token_id])
  started=time.perf_counter()
  cam,prob=inference.explain_answer_token(result,index,'cpu')
  row={'map_seconds':time.perf_counter()-started, 'probability':prob,'min':float(cam.min()),'max':float(cam.max()),'occlusion':{}}
  assert torch.isfinite(cam).all() and cam.any(), 'Invalid or empty map'
  pixels=torch.from_numpy(colormaps['jet'](cam[0].numpy())[...,:3].copy()).permute(2,0,1)
  to_pil_image(pixels).save(out/f'{index}-raw.png')
  overlay_mask(result['input_image'],to_pil_image(cam,mode='F'),alpha=.5).save(out/f'{index}-overlay.png')
  data=inference.answer_inputs(result,index,'cpu')
  data['pixel_values']=data['pixel_values'].detach()
  with torch.no_grad():
   native=model(**data,use_cache=False,logits_to_keep=1).logits[0,-1].float()
   assert abs(float(native.softmax(-1)[token_id])-prob)<1e-6
   logit=float(native[token_id])
   _,_,order,(h,w)=inference.vlm_target(model,data)
   for largest,name in [(True,'top'),(False,'bottom')]:
    chosen=order[cam.flatten().topk(max(1,cam.numel()//5),largest=largest).indices]
    changed={**data,'pixel_values':data['pixel_values'].clone()}
    flat=changed['pixel_values'].reshape(-1,changed['pixel_values'].shape[-1])
    flat[chosen]=0 if model.config.model_type=='qwen3_vl' else .5
    row['occlusion'][name]=logit-float(model(**changed,use_cache=False,logits_to_keep=1).logits[0,-1,token_id])
  report['tokens'][token]=row;save();print(token,json.dumps(row),flush=True)
 assert all(sum(len(m._forward_hooks) for m in model.modules())==0 for _ in [0]),'Leaked hooks'
 save()
except Exception as exc:
 report['error']=str(exc);save();raise

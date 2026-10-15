import gc,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path('demo').resolve()))
import inference
from torchcam.methods import TokenAblation

torch.set_num_threads(2)
out=Path('demo-diagnostics');out.mkdir(exist_ok=True)
report={}
tickets=[
 'I was charged twice for the same order. Please refund the duplicate payment.',
 'My password reset link has expired. Please help me sign in.',
 'My shipment has not arrived. The tracking page shows a delivery delay.',
]
for name in ['minilm','laya']:
 started=time.perf_counter()
 resource,tokenizer,lock=inference.load_resource('decision',name,'cpu')
 report[name]={'load_seconds':time.perf_counter()-started,'cases':[]}
 native=resource.model if name=='laya' else resource
 original=inference.TokenIG
 class Adapter:
  def __init__(self,model,embedding_layer,output_transform=None):
   self.extractor=TokenAblation(model,output_transform)
  def __call__(self,inputs,class_idx=None,*,token_mask=None,**_):
   result=self.extractor(inputs,tokenizer.mask_token_id,token_mask=token_mask)
   maps=result.attributions if class_idx is None else result.attributions[:,[class_idx]]
   return SimpleNamespace(scores=result.scores,attributions=maps)
 inference.TokenIG=Adapter
 for text in tickets:
  count=[0]
  def track(*_):count[0]+=1
  hook=native.register_forward_pre_hook(track)
  started=time.perf_counter()
  try:
   result,maps,offsets,probabilities,target=(inference.explain_laya(resource,text) if name=='laya' else inference.explain_minilm(resource,tokenizer,text))
   best=[]
   for row,spans in zip(maps,offsets):
    positions=sorted(range(len(row)),key=lambda i:row[i],reverse=True)[:5]
    best.append([(text[spans[i][0]:spans[i][1]],row[i]) for i in positions if row[i]>0 and spans[i][1]>spans[i][0]])
   entry={'text':text,'seconds':time.perf_counter()-started,'forwards':count[0],'probabilities':probabilities,'top_tokens':best}
  except Exception as error:
   entry={'text':text,'seconds':time.perf_counter()-started,'error':repr(error)}
  finally:hook.remove()
  report[name]['cases'].append(entry)
  print(json.dumps({'model':name,**entry}),flush=True)
  (out/'report.json').write_text(json.dumps(report,indent=2))
 inference.TokenIG=original
 inference.load_resource.clear();del resource,native;gc.collect()

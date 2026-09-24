from pathlib import Path
import json,sys,torch,numpy as np
root=Path('.');a=torch.load(root/'runs/smoke_crd_seed0_trialstage_b/resume.pt',map_location='cpu',weights_only=False)
b=torch.load(root/'runs/smoke_crd_seed0_trialstage_b_resume/resume.pt',map_location='cpu',weights_only=False)
def same(x,y):
 if isinstance(x,torch.Tensor):torch.testing.assert_close(x,y,rtol=0,atol=0)
 elif isinstance(x,np.ndarray):np.testing.assert_array_equal(x,y)
 elif isinstance(x,dict):
  assert x.keys()==y.keys()
  for k in x:same(x[k],y[k])
 elif isinstance(x,(tuple,list)):
  assert len(x)==len(y)
  for u,v in zip(x,y):same(u,v)
 else:assert x==y,(x,y)
keys=['student','crd','optimizer','rng','epoch','global_step','best']
for k in keys:same(a[k],b[k])
metric_keys=['epoch','global_step','ce','kd','crd','total','train_top1','test_top1','test_top5','test_ce','test_n','lr','train_n','best_top1']
for x,y in zip(a['history'],b['history']):
 for k in metric_keys:same(x[k],y[k])
result=dict(passed=True,comparison='continuous vs stopped at truncated epoch 1 and resumed to truncated epoch 2',exact_state_keys=keys,exact_metrics=metric_keys,global_step=b['global_step'],limitation='workers=0; smoke boundary only; no mid-batch claim')
Path('artifacts/stage_b/real-resume-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

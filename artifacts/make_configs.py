import json
from pathlib import Path
for profile,tau in [('author_code',.07),('paper_tau',.1),('smoke',.07)]:
 for method in ['vanilla','kd','crd','crd_kd']:
  if profile=='paper_tau' and not method.startswith('crd'): continue
  c=dict(profile=profile,method=method,nce_t=tau,nce_k=16384,batch_size=64,workers=0,epochs=2 if profile=='smoke' else 240,max_steps=3 if profile=='smoke' else 0,eval_steps=2 if profile=='smoke' else 0)
  Path(f'configs/{profile}_{method}.json').write_text(json.dumps(c,indent=2))

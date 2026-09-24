import json
from pathlib import Path
for method in ('vanilla','kd','crd','crd_kd'):
 c=json.loads(Path(f'configs/smoke_{method}.json').read_text());c.update(epochs=1,max_steps=30,eval_steps=0)
 Path(f'configs/smoke_benchmark_{method}.json').write_text(json.dumps(c,indent=2))

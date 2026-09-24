import json
from pathlib import Path
c=json.loads(Path('configs/smoke_crd_kd.json').read_text());c.update(epochs=1,max_steps=30,eval_steps=0)
Path('configs/smoke_benchmark_crd_kd.json').write_text(json.dumps(c,indent=2))

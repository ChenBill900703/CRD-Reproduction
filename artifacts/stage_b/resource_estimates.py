import json,hashlib
from pathlib import Path
root=Path('.');rows=[]
for method in ('vanilla','kd','crd','crd_kd'):
 folder=root/f'runs/smoke_{method}_seed0_trialbenchmark30'
 metric=json.loads((folder/'metrics.jsonl').read_text().splitlines()[-1]);status=json.loads((folder/'status.json').read_text())
 assert status['status']=='complete' and metric['global_step']==30 and metric['test_n']==10000
 eval_sec=metric['seconds']-metric['train_seconds'];epoch_sec=metric['train_seconds']/30*782+eval_sec
 rows.append(dict(method=method,train30_seconds=metric['train_seconds'],test10000_seconds=eval_sec,projected_epoch_seconds=epoch_sec,projected_240_hours=epoch_sec*240/3600,peak_allocated_gib=metric['peak_memory_bytes']/2**30,resume_bytes=(folder/'resume.pt').stat().st_size,student_bytes=(folder/'student_final.pt').stat().st_size,final_top1=metric['test_top1'],final_top5=metric['test_top5']))
base=sum(r['projected_240_hours'] for r in rows)*5;tau=sum(r['projected_240_hours'] for r in rows if r['method'].startswith('crd'))*5
out=dict(measured_runs=rows,baseline20_projected_hours=base,paper_tau10_projected_hours=tau,total30_projected_hours=base+tau,planned_buffer_hours=[(base+tau)*1.25,(base+tau)*1.5],formula='(train30_seconds / 30 * 782 + test10000_seconds) * 240 / 3600',caveats=['Extrapolation from 30 steps, NOT measured 240-epoch runtime','Excludes startup and checkpoint I/O','paper_tau throughput assumed equal, not timed on real data','workers=0, FP32; no training setting changed based on test results'])
Path('artifacts/stage_b/resource-estimates.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
# Assert formal results remain empty, every research run except teacher is smoke.
assert json.loads(Path('artifacts/summary.json').read_text())==[]
for folder in Path('runs').iterdir():
 c=json.loads((folder/'config.json').read_text());assert c['profile']=='smoke' or c.get('evaluate_teacher')
print('No formal 240-epoch run started')

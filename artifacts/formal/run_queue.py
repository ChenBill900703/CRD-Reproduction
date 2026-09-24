"""Sequential authorized formal queue; does not alter the frozen training source."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/formal'
def write(state):
 state['updated_at']=time.time();p=OUT/'queue.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));os.replace(tmp,p)
def plan():
 return [(profile,method,seed) for profile,methods in [('author_code',['vanilla','kd','crd','crd_kd']),('paper_tau',['crd','crd_kd'])] for seed in range(5) for method in methods]
def main():
 OUT.mkdir(exist_ok=True)
 if (OUT/'queue.json').exists():raise RuntimeError('Queue already exists; inspect before any restart')
 jobs=plan();state=dict(status='starting',pid=os.getpid(),started_at=time.time(),completed=[],total=len(jobs),current=None)
 write(state)
 try:
  sys.path.insert(0,str(ROOT/'reproduction'));from reproduce import code_hashes,validate_config
  expected=json.loads((ROOT/'artifacts/preflight/code-manifest.json').read_text());assert code_hashes()==expected,'Code changed since preflight'
  for profile,method,seed in jobs:
   cfg=ROOT/f'configs/{profile}_{method}.json';validate_config(json.loads(cfg.read_text()))
   assert not (ROOT/f'runs/{profile}_{method}_seed{seed}_trial1').exists(),'Existing formal run; refusing overwrite'
  for profile,method,seed in jobs:
   run_name=f'{profile}_{method}_seed{seed}_trial1';run=ROOT/'runs'/run_name
   entry='train_teacher.py' if method=='vanilla' else 'train_student.py'
   cmd=[sys.executable,'-u',str(ROOT/'reproduction'/entry),'--config',str(ROOT/f'configs/{profile}_{method}.json'),'--seed',str(seed),'--trial','1']
   log=OUT/f'{run_name}.log'
   with log.open('w',encoding='utf8') as f:
    child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    state.update(status='running',current=dict(name=run_name,pid=child.pid,command=cmd,log=str(log),run_directory=str(run),started_at=time.time()));write(state)
    rc=child.wait()
   result=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
   if rc!=0 or result.get('status')!='complete' or result.get('epoch')!=240:raise RuntimeError(f'{run_name} stopped: exit={rc}, status={result}')
   state['completed'].append(dict(name=run_name,final_top1=result['final_top1'],completed_at=time.time()));write(state)
   subprocess.run([sys.executable,str(ROOT/'reproduction/summarize.py')],cwd=ROOT,check=True)
  subprocess.run([sys.executable,str(ROOT/'reproduction/plot_metrics.py')],cwd=ROOT,check=True)
  state.update(status='complete',current=None,finished_at=time.time());write(state)
 except BaseException as e:
  state.update(status='failed',error=repr(e));write(state);raise
if __name__=='__main__':main()

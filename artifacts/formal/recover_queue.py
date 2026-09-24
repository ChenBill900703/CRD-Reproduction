"""Crash recovery of the authorized queue; frozen training code is untouched."""
import json, os, sys, time, shutil, subprocess
from pathlib import Path
from run_queue import ROOT, OUT, plan, write
sys.path.insert(0,str(ROOT/'reproduction'))
from reproduce import code_hashes, validate_config, seed_all, read_checkpoint, run_lock, sha

def main():
 with run_lock(OUT):
  state=json.loads((OUT/'queue.json').read_text())
  jobs=plan(); names=[f'{p}_{m}_seed{s}_trial1' for p,m,s in jobs]
  done=[r['name'] for r in state['completed']]
  assert done==names[:len(done)]
  assert state['current']['name']==names[len(done)]
  assert code_hashes()==json.loads((ROOT/'artifacts/preflight/code-manifest.json').read_text())
  for r in state['completed']:
   st=json.loads((ROOT/'runs'/r['name']/'status.json').read_text())
   assert st['status']=='complete' and st['epoch']==240 and st['final_top1']==r['final_top1']
  current=Path(state['current']['run_directory'])
  c=json.loads((current/'config.json').read_text());seed_all(c['seed'])
  ck=read_checkpoint(current/'resume.pt',c)
  assert len(ck['history'])==ck['epoch'] and ck['global_step']==ck['epoch']*782
  assert all(r['epoch']==i+1 for i,r in enumerate(ck['history']))
  for p,m,s in jobs[len(done):]:
   cfg=json.loads((ROOT/f'configs/{p}_{m}.json').read_text());validate_config(cfg)
   if s==c['seed'] and p==c['profile'] and m==c['method']:
    assert all(c[k]==v for k,v in cfg.items())
   else: assert not (ROOT/'runs'/f'{p}_{m}_seed{s}_trial1').exists()
  evidence=OUT/('recovery-'+time.strftime('%Y%m%d-%H%M%S'));evidence.mkdir()
  shutil.copytree(current,evidence/'interrupted-run')
  for f in ['queue.json','monitor-state.json','launcher.stdout.log','launcher.stderr.log',current.name+'.log']:
   if (OUT/f).exists():shutil.copy2(OUT/f,evidence/f)
  (evidence/'verification.json').write_text(json.dumps(dict(epoch=ck['epoch'],global_step=ck['global_step'],checkpoint_sha256=sha(current/'resume.pt'),source_config_environment_teacher_validated=True),indent=2))
  print('Verified and backed up checkpoint epoch',ck['epoch'],'at',evidence,flush=True)
  state.update(pid=os.getpid(),recovered_at=time.time(),recovery_evidence=str(evidence))
  try:
   for index,(p,m,s) in enumerate(jobs[len(done):]):
    name=f'{p}_{m}_seed{s}_trial1';run=ROOT/'runs'/name
    entry='train_teacher.py' if m=='vanilla' else 'train_student.py'
    cmd=[sys.executable,'-u',str(ROOT/'reproduction'/entry),'--config',str(ROOT/f'configs/{p}_{m}.json'),'--seed',str(s),'--trial','1']
    if index==0:cmd+=['--resume',str(run/'resume.pt')]
    log=OUT/(name+'.log')
    with log.open('a' if index==0 else 'x',encoding='utf8') as f:
     if index==0:f.write('\n--- Authorized crash recovery from committed epoch ---\n');f.flush()
     child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
     state.update(status='running',current=dict(name=name,pid=child.pid,command=cmd,log=str(log),run_directory=str(run),started_at=time.time()));write(state)
     rc=child.wait()
    result=json.loads((run/'status.json').read_text())
    if rc!=0 or result.get('status')!='complete' or result.get('epoch')!=240:raise RuntimeError(f'{name}: exit={rc}, status={result}')
    state['completed'].append(dict(name=name,final_top1=result['final_top1'],completed_at=time.time()));write(state)
    subprocess.run([sys.executable,str(ROOT/'reproduction/summarize.py')],cwd=ROOT,check=True)
   subprocess.run([sys.executable,str(ROOT/'reproduction/plot_metrics.py')],cwd=ROOT,check=True)
   state.update(status='complete',current=None,finished_at=time.time());write(state)
  except BaseException as e:
   state.update(status='failed',error=repr(e));write(state);raise
if __name__=='__main__':main()

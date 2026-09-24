from pathlib import Path
p=Path('reproduction/reproduce.py');s=p.read_text()
s=s.replace('import argparse, copy, hashlib, json, os, platform, random, subprocess, time','import argparse, hashlib, json, os, platform, random, re, time\nfrom itertools import islice\nfrom contextlib import contextmanager')
s=s.replace("WEIGHTS =", "TEACHER_SHA256 = '22aa12e2b632b19ddde155ccf2388671842ddd304824b3095e0a7fff364beb4f'\nWEIGHTS =")
s=s.replace("cudnn=torch.backends.cudnn.version(),", "cudnn=torch.backends.cudnn.version(), cpu_threads=torch.get_num_threads(),")
s=s.replace("        teacher=model_dict['resnet32x4'](num_classes=100)", "        if sha(c['teacher']) != TEACHER_SHA256: raise ValueError('Teacher checksum differs from audited source')\n        teacher=model_dict['resnet32x4'](num_classes=100)")
s=s.replace("    for i,(x,y) in enumerate(loader):\n        if limit and i>=limit: break", "    for x,y in (islice(loader,limit) if limit else loader):")
s=s.replace("        sums+=np.array([a.item(),b.item(),torch.nn.functional.cross_entropy(z,y).item()])*len(y); n+=len(y)", "        ce=torch.nn.functional.cross_entropy(z,y)\n        if not torch.isfinite(z).all() or not torch.isfinite(ce): raise FloatingPointError('Nonfinite evaluation output/loss')\n        sums+=np.array([a.item(),b.item(),ce.item()])*len(y); n+=len(y)")
s=s.replace("    return dict(top1=float(sums[0]/n)","    if n==0: raise ValueError('Empty evaluation loader')\n    return dict(top1=float(sums[0]/n)")
s=s.replace("    opt.zero_grad(); loss.backward(); opt.step()", "    opt.zero_grad(); loss.backward()\n    gradients=[p.grad for group in opt.param_groups for p in group['params'] if p.grad is not None]\n    if not all(torch.isfinite(grad).all().item() for grad in gradients): raise FloatingPointError('Nonfinite gradient; optimizer not stepped')\n    opt.step()")
start=s.index('def save_checkpoint(');end=s.index('\ndef validate_config',start)
s=s[:start]+'''def atomic_text(path, text):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf8') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)

def atomic_save(path, state):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as f:
        torch.save(state,f); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)

def save_checkpoint(path,s,crd,opt,epoch,global_step,c,g,best,teacher_info,history=None):
    state=dict(schema_version=2,student=s.state_dict(),crd=crd.state_dict() if crd else None,optimizer=opt.state_dict(),epoch=epoch,global_step=global_step,config=c,rng=rng_state(g),best=best,teacher=teacher_info,commit=COMMIT,code_hashes=code_hashes(),environment=environment(),history=history if history is not None else [])
    atomic_save(path,state)

def read_checkpoint(path,c):
    q=torch.load(path,map_location='cpu',weights_only=False) # only our own trusted resume files
    if q.get('schema_version')!=2: raise ValueError('Unsupported checkpoint schema')
    if q['config']!=c: raise ValueError('resume config differs')
    if q['commit']!=COMMIT or q['code_hashes']!=code_hashes(): raise ValueError('resume code differs')
    if q['environment']!=environment(): raise ValueError('resume environment differs')
    if q['teacher'].get('sha256') and q['teacher']['sha256']!=sha(c['teacher']): raise ValueError('teacher differs')
    return q

def restore_checkpoint(q,s,crd,opt,g):
    s.load_state_dict(q['student'])
    if (q['crd'] is None)!=(crd is None): raise ValueError('CRD state differs')
    if crd: crd.load_state_dict(q['crd'])
    opt.load_state_dict(q['optimizer']); restore_rng(q['rng'],g)

def load_checkpoint(path,s,crd,opt,c,g):
    q=read_checkpoint(path,c); restore_checkpoint(q,s,crd,opt,g)
    return q

@contextmanager
def run_lock(out):
    # OS releases the lock even on a killed process; the marker file may remain.
    with (out/'.run.lock').open('a+b') as lock:
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally:
            lock.seek(0)
            if os.name=='nt': msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

def publish_epoch(out,s,c,history,teacher):
    # Rebuild derived files exclusively from the last committed checkpoint history.
    row=history[-1]
    atomic_text(out/'metrics.jsonl',''.join(json.dumps(r,allow_nan=False)+'\\n' for r in history))
    atomic_save(out/'student_final.pt',dict(model=s.state_dict(),epoch=row['epoch'],top1=row['test_top1']))
    status=dict(status='complete' if row['epoch']==c['epochs'] else 'paused',config=str(out/'config.json'),epoch=row['epoch'],final_top1=row['test_top1'],final_top5=row['test_top5'],best_top1=row['best_top1'],elapsed_seconds=sum(r['seconds'] for r in history),peak_memory_bytes=max((r['peak_memory_bytes'] or 0) for r in history),teacher=teacher,checkpoint=str(out/'resume.pt'))
    atomic_text(out/'status.json',json.dumps(status,indent=2,allow_nan=False))
    return status
''' + s[end:]
s=s.replace("    if c['loss_weights']", "    extra_fixed=dict(augmentation=['RandomCrop(32,padding=4)','RandomHorizontalFlip()','ToTensor()','Normalize'],test_augmentation=['ToTensor()','Normalize'],checkpoint_boundary='epoch',seed_list=[0,1,2,3,4],std_ddof=1,workers=0)\n    for key,value in extra_fixed.items():\n        if c.get(key)!=value: raise ValueError(f'Unsupported setting {key}')\n    if c.get('profile') not in ('author_code','paper_tau','smoke'): raise ValueError('Unknown profile')\n    if c.get('method') not in WEIGHTS: raise ValueError('Unknown method')\n    if c['profile']=='paper_tau' and not c['method'].startswith('crd'): raise ValueError('paper_tau requires CRD')\n    for key in ('batch_size','nce_k','epochs'):\n        if type(c.get(key)) is not int or c[key]<=0: raise ValueError(f'Invalid {key}')\n    for key in ('max_steps','eval_steps'):\n        if type(c.get(key)) is not int or c[key]<0: raise ValueError(f'Invalid {key}')\n    if 'seed' in c and (type(c['seed']) is not int or not 0<=c['seed']<2**32): raise ValueError('Invalid seed')\n    if 'trial' in c and not re.fullmatch(r'[A-Za-z0-9_-]+',c['trial']): raise ValueError('Invalid trial filename')\n    if c['loss_weights']")
start=s.index('def main():')
s=s[:start]+'''def main(argv=None, expected_vanilla=None):
    p=argparse.ArgumentParser(description='Scoped CIFAR-100 reproduction; --config is required.')
    p.add_argument('--config',required=True); p.add_argument('--seed',type=int,default=0); p.add_argument('--trial',default='1'); p.add_argument('--resume'); p.add_argument('--evaluate-teacher',action='store_true'); p.add_argument('--download',action='store_true'); p.add_argument('--stop-after-epoch',type=int)
    args=p.parse_args(argv); c=json.loads(Path(args.config).read_text()); c.update(seed=args.seed,trial=args.trial); validate_config(c)
    if expected_vanilla is not None and (c['method']=='vanilla')!=expected_vanilla: raise ValueError('Use train_teacher.py for vanilla and train_student.py for distillation')
    if args.stop_after_epoch is not None and not 1<=args.stop_after_epoch<=c['epochs']: raise ValueError('Invalid stop epoch')
    c['data']=str(ROOT/'data'); c['teacher']=str(ROOT/'assets/resnet32x4_vanilla/ckpt_epoch_240.pth'); c['download']=args.download; c['evaluate_teacher']=args.evaluate_teacher
    seed_all(c['seed'])
    if c['profile']!='smoke' and not torch.cuda.is_available(): raise RuntimeError('Formal FP32 single-GPU run requires CUDA')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); g=torch.Generator().manual_seed(c['seed'])
    out=ROOT/'runs'/f"{c['profile']}_{c['method']}_seed{c['seed']}_trial{c['trial']}"
    if args.evaluate_teacher: out=ROOT/'runs'/'teacher_evaluation'
    if args.resume:
        if args.evaluate_teacher: raise ValueError('Teacher evaluation does not use resume')
        if not (out/'config.json').exists() or json.loads((out/'config.json').read_text())!=c: raise ValueError('Existing run config differs; no files changed')
        if Path(args.resume).resolve()!=(out/'resume.pt').resolve(): raise ValueError('Resume checkpoint must belong to this run')
    else:
        out.mkdir(parents=True,exist_ok=False)
    with run_lock(out):
        # Validate a resume before altering any existing result/config/environment files.
        q=read_checkpoint(args.resume,c) if args.resume else None
        if q and (len(q['history'])!=q['epoch'] or q['history'][-1]['global_step']!=q['global_step']): raise ValueError('Incomplete checkpoint history')
        if q and args.stop_after_epoch is not None and args.stop_after_epoch<=q['epoch']: raise ValueError('Stop epoch already completed')
        if not q:
            atomic_text(out/'config.json',json.dumps(c,indent=2))
            atomic_text(out/'environment.json',json.dumps(environment(),indent=2))
        status={'status':'running','config':str(out/'config.json')}
        def event(kind,**extra):
            with (out/'events.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(dict(event=kind,time=time.time(),**extra))+'\\n')
        event('resume' if q else 'start')
        try:
            train,test=loaders(c,g); s,t,crd,opt=build(c,device)
            ti=q['teacher'] if q else ({'sha256':sha(c['teacher']),'source_url':'http://shape2prog.csail.mit.edu/repo/resnet32x4_vanilla/ckpt_epoch_240.pth','architecture':'resnet32x4'} if t else {})
            if t and not q:
                teacher_begin=time.perf_counter(); ti.update(evaluate(t,test,device)); ti['evaluation_seconds']=time.perf_counter()-teacher_begin
                atomic_text(out/'teacher.json',json.dumps(ti,indent=2)); print('teacher',ti,flush=True)
            if args.evaluate_teacher:
                status.update(status='complete',teacher=ti); return
            history=q['history'] if q else []; global_step=q['global_step'] if q else 0; best=q['best'] if q else 0.
            if q:
                restore_checkpoint(q,s,crd,opt,g)
                status=publish_epoch(out,s,c,history,ti)
                if q['epoch']==c['epochs']: return # Recover interruption after final checkpoint commit.
            sched=SimpleNamespace(learning_rate=.05,lr_decay_epochs=[150,180,210],lr_decay_rate=.1)
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
            for epoch in range(len(history)+1,c['epochs']+1):
                status['status']='running'; atomic_text(out/'status.json',json.dumps(status,indent=2))
                begin=time.perf_counter(); adjust_learning_rate(epoch,sched,opt); sums={}; n=0
                batches=islice(train,c['max_steps']) if c['max_steps'] else train
                for batch in batches:
                    vals,b=step(batch,s,t,crd,opt,c,device); n+=b; global_step+=1
                    for k,v in vals.items(): sums[k]=sums.get(k,0)+v*b
                if not n: raise ValueError('Empty training loader')
                train_seconds=time.perf_counter()-begin
                ev=evaluate(s,test,device,c['eval_steps']); best=max(best,ev['top1'])
                row=dict(epoch=epoch,global_step=global_step,**{k:v/n for k,v in sums.items()},test_top1=ev['top1'],test_top5=ev['top5'],test_ce=ev['ce'],test_n=ev['n'],lr=opt.param_groups[0]['lr'],seconds=time.perf_counter()-begin,train_seconds=train_seconds,train_n=n,best_top1=best,peak_memory_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None)
                history.append(row)
                save_checkpoint(out/'resume.pt',s,crd,opt,epoch,global_step,c,g,best,ti,history)
                status=publish_epoch(out,s,c,history,ti)
                print(row,flush=True)
                if args.stop_after_epoch and epoch>=args.stop_after_epoch: break
        except BaseException as e:
            status.update(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=repr(e)); event('failure',error=repr(e)); raise
        finally: atomic_text(out/'status.json',json.dumps(status,indent=2,allow_nan=False))
if __name__=='__main__': main()
'''
p.write_text(s)
for name,vanilla in [('train_teacher.py',True),('train_student.py',False)]:
 p=Path('reproduction')/name;s=p.read_text();a=s.index('# Reproduction config route:');b=s.index('import tensorboard_logger as tb_logger',a)
 s=s[:a]+f'''# Executable route is intentionally scoped; original implementation is preserved below.
if __name__ == '__main__':
    from reproduce import main
    main(expected_vanilla={vanilla})
    raise SystemExit(0)

'''+s[b:];p.write_text(s)

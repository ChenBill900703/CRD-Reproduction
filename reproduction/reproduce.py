"""Scoped CIFAR-100 runner; official models, sampler and losses remain authoritative."""
import argparse, hashlib, json, os, platform, random, re, time
from itertools import islice
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import model_dict
from dataset.cifar100 import CIFAR100InstanceSample
from crd.criterion import CRDLoss
from distiller_zoo.KD import DistillKL
from helper.util import adjust_learning_rate, accuracy

ROOT = Path(__file__).resolve().parent.parent
COMMIT = 'b84f547c5db6a35318d4671d7d5c4de74c822403'
TEACHER_SHA256 = '22aa12e2b632b19ddde155ccf2388671842ddd304824b3095e0a7fff364beb4f'
WEIGHTS = {'vanilla': (1,0,0), 'kd': (.1,.9,0), 'crd': (1,0,.8), 'crd_kd': (1,1,.8)}
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def code_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'reproduction').rglob('*.py'))}
def environment():
    import torchvision
    return dict(python=platform.python_version(), os=platform.platform(), torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None, cudnn=torch.backends.cudnn.version(), cpu_threads=torch.get_num_threads(), benchmark=torch.backends.cudnn.benchmark, deterministic=torch.backends.cudnn.deterministic, tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32)
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)
def worker_seed(worker):
    s=torch.initial_seed()%2**32; np.random.seed(s); random.seed(s)
def rng_state(g):
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],loader=g.get_state())
def restore_rng(s,g):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['cpu']); g.set_state(s['loader'])
    if s['cuda']: torch.cuda.set_rng_state_all(s['cuda'])
def loaders(c,g):
    norm=transforms.Normalize((.5071,.4867,.4408),(.2675,.2565,.2761))
    tr=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),norm])
    te=transforms.Compose([transforms.ToTensor(),norm])
    kw=dict(root=c['data'],train=True,download=c.get('download',False),transform=tr)
    train=CIFAR100InstanceSample(**kw,k=c['nce_k'],mode='exact',percent=1.) if c['method'].startswith('crd') else datasets.CIFAR100(**kw)
    test=datasets.CIFAR100(root=c['data'],train=False,download=c.get('download',False),transform=te)
    if len(train)!=50000 or len(test)!=10000: raise ValueError('Unexpected CIFAR-100 split sizes')
    return (DataLoader(train,batch_size=c['batch_size'],shuffle=True,num_workers=c['workers'],drop_last=False,generator=g,worker_init_fn=worker_seed,persistent_workers=False),DataLoader(test,batch_size=32,shuffle=False,num_workers=c['workers']//2,generator=g,worker_init_fn=worker_seed,persistent_workers=False))
def build(c,device):
    teacher=None
    if c['method']!='vanilla' or c.get('evaluate_teacher'):
        if sha(c['teacher']) != TEACHER_SHA256: raise ValueError('Teacher checksum differs from audited source')
        teacher=model_dict['resnet32x4'](num_classes=100)
        # Official file is provenance-checked and contains tensors/optimizer state.
        with torch.serialization.safe_globals([(np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.dtype, np.dtypes.Float64DType]):
            ck=torch.load(c['teacher'],map_location='cpu',weights_only=True)
        if ck.get('epoch') != 240: raise ValueError('Teacher is not epoch 240')
        teacher.load_state_dict(ck['model'],strict=True); teacher.to(device).eval().requires_grad_(False)
    student=model_dict['resnet8x4'](num_classes=100).to(device)
    crd=CRDLoss(SimpleNamespace(s_dim=256,t_dim=256,feat_dim=128,n_data=50000,nce_k=c['nce_k'],nce_t=c['nce_t'],nce_m=.5)).to(device) if c['method'].startswith('crd') else None
    params=list(student.parameters())+(list(crd.parameters()) if crd else [])
    optimizer=torch.optim.SGD(params,lr=.05,momentum=.9,weight_decay=.0005)
    return student,teacher,crd,optimizer
@torch.no_grad()
def evaluate(model,loader,device,limit=0):
    model.eval(); n=0; sums=np.zeros(3)
    for x,y in (islice(loader,limit) if limit else loader):
        x,y=x.to(device),y.to(device); z=model(x); a,b=accuracy(z,y,(1,5))
        ce=torch.nn.functional.cross_entropy(z,y)
        if not torch.isfinite(z).all() or not torch.isfinite(ce): raise FloatingPointError('Nonfinite evaluation output/loss')
        sums+=np.array([a.item(),b.item(),ce.item()])*len(y); n+=len(y)
    if n==0: raise ValueError('Empty evaluation loader')
    return dict(top1=float(sums[0]/n),top5=float(sums[1]/n),ce=float(sums[2]/n),n=n)
def step(batch,s,t,crd,opt,c,device):
    s.train()
    x,y=batch[0].to(device),batch[1].to(device)
    fs,z=s(x,is_feat=True); ce=torch.nn.functional.cross_entropy(z,y); kd=z.new_zeros(()); cl=z.new_zeros(())
    if t is not None:
        t.eval()
        with torch.no_grad(): ft,zt=t(x,is_feat=True)
        kd=DistillKL(4)(z,zt)
        if crd is not None: cl=crd(fs[-1],ft[-1],batch[2].to(device),batch[3].to(device)).squeeze()
    r,a,b=WEIGHTS[c['method']]; loss=r*ce+a*kd+b*cl
    if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
    opt.zero_grad(); loss.backward()
    gradients=[p.grad for group in opt.param_groups for p in group['params'] if p.grad is not None]
    if not torch.stack([torch.isfinite(grad).all() for grad in gradients]).all().item(): raise FloatingPointError('Nonfinite gradient; optimizer not stepped')
    opt.step()
    if not torch.stack([torch.isfinite(p).all() for group in opt.param_groups for p in group['params']]).all().item(): raise FloatingPointError('Nonfinite parameter; discard step and resume last committed epoch')
    acc=accuracy(z,y,(1,))[0].item()
    return dict(ce=ce.item(),kd=kd.item(),crd=cl.item(),total=loss.item(),train_top1=acc),len(y)
def atomic_text(path, text):
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
    atomic_text(out/'metrics.jsonl',''.join(json.dumps(r,allow_nan=False)+'\n' for r in history))
    atomic_save(out/'student_final.pt',dict(model=s.state_dict(),epoch=row['epoch'],top1=row['test_top1']))
    status=dict(global_step=row['global_step'],train_n=row['train_n'],test_n=row['test_n'],code_fingerprint=hashlib.sha256(json.dumps(code_hashes(),sort_keys=True).encode()).hexdigest(),status='complete' if row['epoch']==c['epochs'] else 'paused',config=str(out/'config.json'),epoch=row['epoch'],final_top1=row['test_top1'],final_top5=row['test_top5'],best_top1=row['best_top1'],elapsed_seconds=sum(r['seconds'] for r in history),peak_memory_bytes=max((r['peak_memory_bytes'] or 0) for r in history),teacher=teacher,checkpoint=str(out/'resume.pt'))
    atomic_text(out/'status.json',json.dumps(status,indent=2,allow_nan=False))
    return status

def validate_config(c):
    fixed=dict(model_s='resnet8x4',model_t='resnet32x4',dataset='cifar100',test_batch_size=32,shuffle=True,drop_last=False,mode='exact',percent=1.0,feat_dim=128,n_data=50000,nce_m=.5,kd_T=4,learning_rate=.05,momentum=.9,weight_decay=.0005,lr_decay_epochs=[150,180,210],lr_decay_rate=.1,precision='fp32',tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,normalization_mean=[.5071,.4867,.4408],normalization_std=[.2675,.2565,.2761],optimizer='SGD',nesterov=False,dampening=0,num_gpus=1)
    for key,value in fixed.items():
        if c.get(key)!=value: raise ValueError(f'Unsupported setting {key}: {c.get(key)}')
    extra_fixed=dict(augmentation=['RandomCrop(32,padding=4)','RandomHorizontalFlip()','ToTensor()','Normalize'],test_augmentation=['ToTensor()','Normalize'],checkpoint_boundary='epoch',seed_list=[0,1,2,3,4],std_ddof=1,workers=0)
    for key,value in extra_fixed.items():
        if c.get(key)!=value: raise ValueError(f'Unsupported setting {key}')
    allowed=set(fixed)|set(extra_fixed)|{'profile','method','epochs','batch_size','nce_k','nce_t','loss_weights','max_steps','eval_steps','seed','trial','data','teacher','download','evaluate_teacher'}
    unknown=set(c)-allowed
    if unknown: raise ValueError(f'Unknown config fields: {sorted(unknown)}')
    if c.get('profile') not in ('author_code','paper_tau','smoke'): raise ValueError('Unknown profile')
    if c.get('method') not in WEIGHTS: raise ValueError('Unknown method')
    if c['profile']=='paper_tau' and not c['method'].startswith('crd'): raise ValueError('paper_tau requires CRD')
    for key in ('batch_size','nce_k','epochs'):
        if type(c.get(key)) is not int or c[key]<=0: raise ValueError(f'Invalid {key}')
    for key in ('max_steps','eval_steps'):
        if type(c.get(key)) is not int or c[key]<0: raise ValueError(f'Invalid {key}')
    if 'seed' in c and (type(c['seed']) is not int or not 0<=c['seed']<2**32): raise ValueError('Invalid seed')
    if c['profile']!='smoke' and 'seed' in c and c['seed'] not in c['seed_list']: raise ValueError('Formal seed not in preregistered list')
    if 'trial' in c and not re.fullmatch(r'[A-Za-z0-9_-]+',c['trial']): raise ValueError('Invalid trial filename')
    if c['loss_weights']!=list(WEIGHTS[c['method']]): raise ValueError('Loss weights differ')
    if c['nce_t'] != (.1 if c['profile']=='paper_tau' else .07): raise ValueError('Temperature profile differs')
    if c['profile']!='smoke' and (c['epochs']!=240 or c['batch_size']!=64 or c['nce_k']!=16384 or c.get('max_steps') or c.get('eval_steps')): raise ValueError('Formal config must be full benchmark')

def main(argv=None, expected_vanilla=None):
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
            with (out/'events.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(dict(event=kind,time=time.time(),**extra))+'\n')
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

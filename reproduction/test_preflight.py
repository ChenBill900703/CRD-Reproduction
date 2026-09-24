"""No CIFAR files or downloads: synthetic fixtures exercise production execution paths."""
import copy, gc, importlib, io, json, shutil, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import reproduce as r

BASE=r.ROOT
class Synthetic(Dataset):
    def __init__(self,n,contrast=False): self.n=n; self.contrast=contrast; self.accesses=0
    def __len__(self): return self.n
    def __getitem__(self,i):
        self.accesses+=1
        x=torch.rand(3,32,32)+float(np.random.rand())*.01
        if self.contrast: return x,i%100,i,np.r_[i,np.random.choice(np.arange(10,100),32,replace=False)]
        return x,i%100

def synthetic_loaders(c,g):
    return (DataLoader(Synthetic(5,c['method'].startswith('crd')),batch_size=2,shuffle=True,generator=g),DataLoader(Synthetic(3),batch_size=2,shuffle=False,generator=g))

def config(method='crd_kd'):
    c=json.loads((BASE/f'configs/smoke_{method}.json').read_text())
    c.update(batch_size=2,nce_k=32,epochs=2,max_steps=0,eval_steps=0)
    return c

def assert_tree(test,a,b):
    if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        test.assertEqual(a.keys(),b.keys())
        for k in a: assert_tree(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b): assert_tree(test,x,y)
    else: test.assertEqual(a,b)

class Preflight(unittest.TestCase):
    def setUp(self): r.seed_all(123)
    def test_config_rejects_ignored_and_invalid_settings(self):
        for p in (BASE/'configs').glob('*.json'): r.validate_config(json.loads(p.read_text()))
        for field,value in [('warmup',True),('profile','typo'),('method','typo'),('workers',8),('epochs',0),('max_steps',-1),('eval_steps',-1),('batch_size',0),('nce_k',0),('trial','../bad'),('seed',-1),('augmentation',[]),('test_augmentation',[]),('normalization_std',[1,1,1])]:
            with self.subTest(field=field):
                c=config();c[field]=value
                with self.assertRaises(ValueError):r.validate_config(c)
        c=json.loads((BASE/'configs/author_code_crd.json').read_text());c['max_steps']=1
        with self.assertRaises(ValueError):r.validate_config(c)
    def test_cli_help_and_wrong_entrypoint(self):
        for entry in ['train_teacher.py','train_student.py','reproduce.py']:
            p=subprocess.run([sys.executable,str(BASE/'reproduction'/entry),'--help'],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr);self.assertIn('--seed',p.stdout)
        p=subprocess.run([sys.executable,str(BASE/'reproduction/train_teacher.py'),'--config='+str(BASE/'configs/smoke_crd.json')],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('Use train_teacher.py',p.stderr)
    def test_synthetic_official_sampler_and_loader(self):
        def fake_init(ds,root,train=True,transform=None,target_transform=None,download=False):
            ds.train=train;ds.transform=transform;ds.target_transform=target_transform
            ds.data=np.broadcast_to(np.arange(3072,dtype=np.uint8).reshape(1,32,32,3),(50000 if train else 10000,32,32,3))
            ds.targets=(np.arange(len(ds.data))%100).tolist()
        c=json.loads((BASE/'configs/smoke_crd.json').read_text());c['data']='MUST_NOT_BE_READ'
        with patch.object(r.datasets.CIFAR100,'__init__',fake_init):
            train,test=r.loaders(c,torch.Generator().manual_seed(3))
            self.assertEqual(len(train),782);self.assertFalse(train.drop_last);self.assertEqual(test.batch_size,32)
            self.assertEqual(train.dataset.cls_negative.shape,(100,49500))
            batch=next(iter(train));idx=batch[3];labels=np.asarray(train.dataset.targets)
            self.assertEqual(idx.shape,(64,16385));self.assertTrue(torch.equal(batch[2],idx[:,0]))
            self.assertTrue(np.all(labels[idx[:,1:].numpy()]!=batch[1].numpy()[:,None]))
            self.assertTrue(all(len(np.unique(row))==16384 for row in idx[:,1:].numpy()))
            self.assertEqual(train.dataset[49999][2],49999)
            tr=train.dataset.transform.transforms
            self.assertEqual(tr[0].size,(32,32));self.assertEqual(tr[0].padding,4)
            self.assertEqual(tr[-1].std,(.2675,.2565,.2761))
            self.assertEqual(len(test.dataset.transform.transforms),2)
    def test_evaluate_empty_nonfinite_weighting_and_limit(self):
        class Identity(torch.nn.Module):
            def forward(self,x):return x
        z=torch.tensor([[9.,0,0,0,0,0],[0,9.,0,0,0,0],[9.,0,0,0,0,0]])
        loader=DataLoader(torch.utils.data.TensorDataset(z,torch.tensor([0,1,1])),batch_size=2)
        result=r.evaluate(Identity(),loader,'cpu');self.assertAlmostEqual(result['top1'],200/3);self.assertEqual(result['n'],3)
        self.assertEqual(r.evaluate(Identity(),loader,'cpu',1)['n'],2)
        with self.assertRaises(ValueError):r.evaluate(Identity(),[],'cpu')
        z[0,0]=float('nan')
        with self.assertRaises(FloatingPointError):r.evaluate(Identity(),loader,'cpu')
        class Counting:
            def __init__(self):self.n=0
            def __iter__(self):
                for i in range(4):self.n+=1;yield torch.ones(2,6),torch.zeros(2,dtype=torch.long)
        ds=Counting();r.evaluate(Identity(),ds,'cpu',1);self.assertEqual(ds.n,1)
    def test_atomic_failure_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=BASE/'artifacts/preflight') as tmp:
            path=Path(tmp)/'state.pt';r.atomic_save(path,{'value':1});before=path.read_bytes()
            with patch.object(r.os,'replace',side_effect=OSError('simulated disk error')):
                with self.assertRaises(OSError):r.atomic_save(path,{'value':2})
            self.assertEqual(path.read_bytes(),before)
    def test_exclusive_run_lock(self):
        with tempfile.TemporaryDirectory(dir=BASE/'artifacts/preflight') as tmp:
            with r.run_lock(Path(tmp)):
                with self.assertRaises(OSError):
                    with r.run_lock(Path(tmp)):pass
            with r.run_lock(Path(tmp)):pass
    def test_all_methods_main_resume_and_final_commit_recovery(self):
        # Use actual models/official teacher/optimizer/CRD, replacing data only.
        device='cuda' if torch.cuda.is_available() else 'cpu'
        for method in r.WEIGHTS:
            with self.subTest(method=method),tempfile.TemporaryDirectory(dir=BASE/'artifacts/preflight') as tmp:
                root=Path(tmp);teacher=root/'assets/resnet32x4_vanilla/ckpt_epoch_240.pth';teacher.parent.mkdir(parents=True)
                shutil.copyfile(BASE/'assets/resnet32x4_vanilla/ckpt_epoch_240.pth',teacher)
                cfg=root/'config.json';cfg.write_text(json.dumps(config(method)))
                with patch.object(r,'ROOT',root),patch.object(r,'loaders',synthetic_loaders),redirect_stdout(io.StringIO()):
                    r.main(['--config',str(cfg),'--trial','continuous'])
                    r.main(['--config',str(cfg),'--trial','resumed','--stop-after-epoch','1'])
                    out=root/'runs'/f'smoke_{method}_seed0_trialresumed'
                    # Simulate partial metrics/status and stale inference file after a crash.
                    (out/'metrics.jsonl').write_text('{torn-json');(out/'status.json').write_text('{torn-json')
                    r.main(['--config',str(cfg),'--trial','resumed','--resume',str(out/'resume.pt')])
                    a=torch.load(root/'runs'/f'smoke_{method}_seed0_trialcontinuous/resume.pt',weights_only=False,map_location='cpu')
                    b=torch.load(out/'resume.pt',weights_only=False,map_location='cpu')
                    for key in ['student','crd','optimizer','rng','global_step','best']:assert_tree(self,a[key],b[key])
                    self.assertEqual(b['global_step'],6);self.assertEqual(b['history'][-1]['train_n'],5)
                    self.assertEqual(json.loads((out/'status.json').read_text())['status'],'complete')
                    # Last checkpoint committed but publication interrupted: no extra training allowed.
                    (out/'metrics.jsonl').write_text('broken');(out/'status.json').write_text('broken');(out/'student_final.pt').write_bytes(b'broken')
                    with patch.object(r,'step',side_effect=AssertionError('must not train completed checkpoint')):
                        r.main(['--config',str(cfg),'--trial','resumed','--resume',str(out/'resume.pt')])
                    self.assertEqual(len((out/'metrics.jsonl').read_text().splitlines()),2)
                    self.assertEqual(torch.load(out/'student_final.pt',weights_only=True)['epoch'],2)
                    before=(out/'status.json').read_bytes()
                    with patch.object(r,'code_hashes',return_value={'changed':'yes'}):
                        with self.assertRaises(ValueError):r.main(['--config',str(cfg),'--trial','resumed','--resume',str(out/'resume.pt')])
                    self.assertEqual(before,(out/'status.json').read_bytes())
    def test_nonfinite_gradient_stops_before_optimizer(self):
        c=config('vanilla');s=r.model_dict['resnet8x4'](num_classes=100);opt=torch.optim.SGD(s.parameters(),lr=.05)
        before=s.fc.weight.clone();handle=s.fc.weight.register_hook(lambda grad:torch.full_like(grad,float('inf')))
        with self.assertRaises(FloatingPointError):r.step((torch.randn(2,3,32,32),torch.tensor([0,1])),s,None,None,opt,c,'cpu')
        torch.testing.assert_close(before,s.fc.weight,rtol=0,atol=0);handle.remove()
    def test_cpu_all_methods_step(self):
        for method in r.WEIGHTS:
            c=config(method);c['teacher']=str(BASE/'assets/resnet32x4_vanilla/ckpt_epoch_240.pth')
            s,t,crd,opt=r.build(c,'cpu');batch=next(iter(synthetic_loaders(c,torch.Generator())[0]))
            vals,n=r.step(batch,s,t,crd,opt,c,'cpu');self.assertEqual(n,2);self.assertTrue(all(np.isfinite(list(vals.values()))))

    def test_plot_synthetic_metrics(self):
        from plot_metrics import plot_runs
        root=BASE/'artifacts/preflight/plot-fixture';folder=root/'synthetic_fixture_NOT_RESEARCH_RESULTS';folder.mkdir(parents=True,exist_ok=True)
        (folder/'config.json').write_text(json.dumps({'max_steps':1}))
        rows=[dict(epoch=i,ce=4.6,kd=.2,crd=10.,total=12.8,train_top1=1.,test_top1=2.,test_top5=5.,lr=.05) for i in (1,2)]
        (folder/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        paths=plot_runs(root);self.assertEqual(len(paths),1)
        from PIL import Image
        with Image.open(paths[0]) as im:self.assertEqual(im.size,(1950,525))

    def test_full_crd_upstream_parity_both_temperatures(self):
        import types
        package=types.ModuleType('frozen_crd');package.__path__=[str(BASE/'reference/RepDistiller/crd')]
        sys.modules['frozen_crd']=package
        original_memory=importlib.import_module('frozen_crd.memory')
        original_criterion=importlib.import_module('frozen_crd.criterion')
        from types import SimpleNamespace
        for tau in (.07,.1):
            opt=SimpleNamespace(s_dim=256,t_dim=256,feat_dim=128,n_data=100,nce_k=16,nce_t=tau,nce_m=.5)
            torch.manual_seed(17)
            with patch.object(original_memory.AliasMethod,'cuda',lambda self:None):a=original_criterion.CRDLoss(opt)
            torch.manual_seed(17);b=r.CRDLoss(opt)
            assert_tree(self,a.state_dict(),b.state_dict())
            fs=torch.randn(4,256,requires_grad=True);ft=torch.randn(4,256,requires_grad=True)
            fs2=fs.detach().clone().requires_grad_();ft2=ft.detach().clone().requires_grad_()
            idx=torch.arange(4);contrast=torch.cat([idx[:,None],torch.randint(4,100,(4,16))],1)
            la=a(fs,ft,idx,contrast);lb=b(fs2,ft2,idx,contrast)
            torch.testing.assert_close(la,lb,rtol=0,atol=0);la.backward();lb.backward()
            for x,y in [(fs.grad,fs2.grad),(ft.grad,ft2.grad)]:torch.testing.assert_close(x,y,rtol=0,atol=0)
            for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)
            assert_tree(self,a.state_dict(),b.state_dict())
    @unittest.skipUnless(torch.cuda.is_available(),'Requires CUDA for formal-size stress test')
    def test_gpu_formal_shape_multistep_both_tau(self):
        results=[]
        for tau in (.07,.1):
            gc.collect(); torch.cuda.empty_cache() # Isolate unreachable tensors from preceding test fixtures.
            c=config();c.update(nce_k=16384,nce_t=tau,teacher=str(BASE/'assets/resnet32x4_vanilla/ckpt_epoch_240.pth'))
            s,t,crd,opt=r.build(c,'cuda');teacher={k:v.clone() for k,v in t.state_dict().items()}
            ptrs=[]
            hs=s.register_forward_pre_hook(lambda mod,args:ptrs.append(args[0].data_ptr()))
            ht=t.register_forward_pre_hook(lambda mod,args:ptrs.append(args[0].data_ptr()))
            torch.cuda.reset_peak_memory_stats();begin=r.time.perf_counter();allocated=[];losses=[]
            for i in range(24):
                batch=(torch.randn(64,3,32,32),torch.arange(64),torch.arange(64),torch.cat([torch.arange(64)[:,None],torch.randint(64,50000,(64,16384))],1))
                vals,_=r.step(batch,s,t,crd,opt,c,'cuda');losses.append(vals['total']);allocated.append(torch.cuda.memory_allocated())
                self.assertEqual(ptrs[-2],ptrs[-1])
            torch.cuda.synchronize();seconds=r.time.perf_counter()-begin
            print('GPU allocation trace',tau,allocated,flush=True)
            (BASE/'artifacts/preflight/allocation-trace.json').write_text(json.dumps(dict(tau=tau,allocated=allocated),indent=2))
            self.assertLessEqual(max(allocated[2:])-min(allocated[2:]),1024*1024)
            self.assertTrue(all(torch.isfinite(v).all() for v in crd.state_dict().values()))
            assert_tree(self,teacher,t.state_dict());hs.remove();ht.remove()
            results.append(dict(tau=tau,steps=24,batch=64,K=16384,n_data=50000,seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),steady_allocated_growth_bytes=max(allocated[2:])-min(allocated[2:]),loss_first=losses[0],loss_last=losses[-1],data='synthetic; NOT benchmark accuracy'))
            del s,t,crd,opt,teacher;torch.cuda.empty_cache()
        (BASE/'artifacts/preflight/gpu-stress.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':unittest.main(verbosity=2)

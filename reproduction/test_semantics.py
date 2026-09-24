"""Semantic regression tests against frozen upstream, not just shape smoke tests."""
import copy, importlib.util, json, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from reproduce import *

class Semantics(unittest.TestCase):
 def setUp(self): seed_all(123)
 def test_original_numerical_parity(self):
  # Only neutralize upstream constructor's device call; all upstream math untouched.
  spec=importlib.util.spec_from_file_location('upstream_memory',ROOT/'reference/RepDistiller/crd/memory.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
  with patch.object(mod.AliasMethod,'cuda',lambda self:None): original=mod.ContrastMemory(8,23,5)
  from crd.memory import ContrastMemory
  new=ContrastMemory(8,23,5)
  self.assertLessEqual(new.memory_v1.abs().max().item(),(3/8)**.5)
  self.assertFalse(torch.allclose(new.memory_v1.norm(dim=1),torch.ones(23)))
  new.load_state_dict(original.state_dict())
  from crd.criterion import ContrastLoss
  a=F.normalize(torch.randn(3,8),dim=1).requires_grad_(); b=F.normalize(torch.randn(3,8),dim=1).requires_grad_(); aa=a.detach().clone().requires_grad_(); bb=b.detach().clone().requires_grad_()
  y=torch.tensor([2,7,19]); idx=torch.randint(0,23,(3,6)); idx[:,0]=y
  before=new.memory_v1.clone(); outs=original(a,b,y,idx); outn=new(aa,bb,y,idx)
  lo=sum(ContrastLoss(23)(v) for v in outs); ln=sum(ContrastLoss(23)(v) for v in outn)
  torch.testing.assert_close(lo,ln,rtol=0,atol=0); lo.backward(); ln.backward()
  torch.testing.assert_close(a.grad,aa.grad,rtol=0,atol=0); torch.testing.assert_close(b.grad,bb.grad,rtol=0,atol=0)
  for key,v in original.state_dict().items(): torch.testing.assert_close(v,new.state_dict()[key],rtol=0,atol=0)
  untouched=torch.ones(23,dtype=torch.bool); untouched[y]=False
  torch.testing.assert_close(before[untouched],new.memory_v1[untouched],rtol=0,atol=0)
  torch.testing.assert_close(new.memory_v1[y],F.normalize(.5*before[y]+.5*aa.detach(),dim=1))
  z=new.params[2:4].clone(); new(aa.detach(),bb.detach(),y,idx); torch.testing.assert_close(z,new.params[2:4],rtol=0,atol=0)
  # Independent scalar formula catches accidental K or direction averaging.
  x=outn[0].detach(); noise=5/23; expected=-(torch.log(x[:,0]/(x[:,0]+noise+1e-7)).sum()+torch.log(noise/(x[:,1:]+noise+1e-7)).sum())/3
  torch.testing.assert_close(ContrastLoss(23)(x).squeeze(),expected)
 def test_kd_and_lr(self):
  a=torch.randn(7,100,requires_grad=True); b=torch.randn(7,100)
  expected=F.kl_div(F.log_softmax(a/4,1),F.softmax(b/4,1),reduction='batchmean')*16
  torch.testing.assert_close(DistillKL(4)(a,b),expected)
  opt=torch.optim.SGD([a],lr=.05); cfg=SimpleNamespace(learning_rate=.05,lr_decay_epochs=[150,180,210],lr_decay_rate=.1)
  for epoch,lr in [(150,.05),(151,.005),(180,.005),(181,.0005),(210,.0005),(211,.00005)]:
   adjust_learning_rate(epoch,cfg,opt); self.assertAlmostEqual(opt.param_groups[0]['lr'],lr)
 def test_alias_cpu(self):
  from crd.memory import ContrastMemory
  m=ContrastMemory(8,20,3); y=torch.tensor([1,5]); a=F.normalize(torch.randn(2,8),dim=1)
  self.assertEqual(m(a,a,y)[0].shape,(2,4,1))
 def test_sampler_real(self):
  d=CIFAR100InstanceSample(ROOT/'data',train=True,download=False,k=16384,transform=transforms.ToTensor())
  self.assertEqual(len(d),50000); labels=np.asarray(d.targets)
  rows=[]
  for i in [0,49999,1327]:
   x,y,j,idx=d[i]; self.assertEqual(j,i); self.assertEqual(idx[0],i); self.assertEqual(len(set(idx[1:])),16384); self.assertTrue(np.all(labels[idx[1:]]!=y)); rows.append(idx)
  self.assertEqual(np.stack(rows).shape,(3,16385)); self.assertEqual(d.cls_negative.shape,(100,49500))
  batch=next(iter(DataLoader(d,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(1))))
  self.assertEqual(batch[3].shape,(64,16385)); self.assertTrue(torch.equal(batch[2],batch[3][:,0]))
  self.assertTrue(np.all(labels[batch[3][:,1:].numpy()]!=batch[1].numpy()[:,None]))
 def test_full_step_and_resume(self):
  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  c=json.loads((ROOT/'configs/author_code_crd_kd.json').read_text()); c.update(teacher=str(ROOT/'assets/resnet32x4_vanilla/ckpt_epoch_240.pth'),seed=123)
  s,t,crd,opt=build(c,device); g=torch.Generator().manual_seed(123)
  oldt={k:v.clone() for k,v in t.state_dict().items()}; olds=s.fc.weight.detach().clone(); oldheads=[crd.embed_s.linear.weight.detach().clone(),crd.embed_t.linear.weight.detach().clone()]
  x=torch.randn(64,3,32,32); y=torch.arange(64)%100; idx=torch.arange(64); negatives=torch.randint(64,50000,(64,16384)); ci=torch.cat([idx[:,None],negatives],1)
  with torch.no_grad():
   fs,z=s(x.to(device),is_feat=True); ft,zt=t(x.to(device),is_feat=True)
   self.assertEqual(fs[-1].shape,(64,256)); self.assertEqual(ft[-1].shape,(64,256)); self.assertEqual(z.shape,(64,100)); self.assertEqual(crd.embed_s(fs[-1]).shape,(64,128))
  if device.type=='cuda': torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
  begin=time.perf_counter(); vals,_=step((x,y,idx,ci),s,t,crd,opt,c,device)
  if device.type=='cuda': torch.cuda.synchronize()
  result=dict(device=str(device),seconds=time.perf_counter()-begin,peak_memory_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,losses=vals,batch=64,K=16384,n_data=50000)
  (ROOT/'artifacts/full_step.json').write_text(json.dumps(result,indent=2)); print(result,flush=True)
  for k,v in t.state_dict().items(): torch.testing.assert_close(v,oldt[k],rtol=0,atol=0)
  self.assertFalse(torch.equal(olds,s.fc.weight)); self.assertFalse(torch.equal(oldheads[0],crd.embed_s.linear.weight)); self.assertFalse(torch.equal(oldheads[1],crd.embed_t.linear.weight)); self.assertTrue(all(p.grad is None for p in t.parameters()))
  # Full checkpoint with memories, Z, projections, optimizer and all RNG states.
  with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
   path=Path(tmp)/'resume.pt'; save_checkpoint(path,s,crd,opt,1,1,c,g,0.,{'sha256':sha(c['teacher'])})
   expected_rng=(random.random(),np.random.rand(),torch.rand(3),torch.rand(3,generator=g))
   state={k:v.detach().clone() for k,v in crd.state_dict().items()}
   q=load_checkpoint(path,s,crd,opt,c,g)
   self.assertEqual(q['epoch'],1); actual_rng=(random.random(),np.random.rand(),torch.rand(3),torch.rand(3,generator=g))
   for a,b in zip(expected_rng,actual_rng): np.testing.assert_array_equal(a,b)
   for k,v in crd.state_dict().items(): torch.testing.assert_close(v,state[k],rtol=0,atol=0)
   expected,_=step((x,y,idx,ci),s,t,crd,opt,c,device); expected_weights=s.fc.weight.detach().clone(); expected_mem=crd.contrast.memory_v1.clone()
   load_checkpoint(path,s,crd,opt,c,g); actual,_=step((x,y,idx,ci),s,t,crd,opt,c,device)
   self.assertEqual(expected,actual); torch.testing.assert_close(s.fc.weight,expected_weights,rtol=0,atol=0); torch.testing.assert_close(crd.contrast.memory_v1,expected_mem,rtol=0,atol=0)
 def test_epoch_boundary_loader_resume(self):
  # Exhausted epoch with shuffle, random crop/flip and NumPy sampling, workers=0.
  class Tiny(torch.utils.data.Dataset):
   def __len__(self): return 16
   def __getitem__(self,i):
    x=torch.rand(3,32,32)+random.random()+float(np.random.rand())
    return x,i%100,i,np.r_[i,np.random.choice(np.arange(16,100),32,replace=False)]
  c=json.loads((ROOT/'configs/smoke_crd.json').read_text()); c.update(teacher=str(ROOT/'assets/resnet32x4_vanilla/ckpt_epoch_240.pth'),nce_k=32,seed=123)
  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); s,t,crd,opt=build(c,device); g=torch.Generator().manual_seed(123)
  loader=DataLoader(Tiny(),batch_size=8,shuffle=True,generator=g,num_workers=0)
  for b in loader: step(b,s,t,crd,opt,c,device)
  with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
   path=Path(tmp)/'resume.pt'; save_checkpoint(path,s,crd,opt,1,2,c,g,0.,{'sha256':sha(c['teacher'])})
   expected=[]
   for b in loader: expected.append((b[2].tolist(),step(b,s,t,crd,opt,c,device)[0]))
   state={k:v.clone() for k,v in s.state_dict().items()}; memories={k:v.clone() for k,v in crd.state_dict().items()}
   s2,t2,crd2,opt2=build(c,device); g2=torch.Generator(); load_checkpoint(path,s2,crd2,opt2,c,g2)
   actual=[]
   for b in DataLoader(Tiny(),batch_size=8,shuffle=True,generator=g2,num_workers=0): actual.append((b[2].tolist(),step(b,s2,t2,crd2,opt2,c,device)[0]))
   self.assertEqual(expected,actual)
   for k,v in s2.state_dict().items(): torch.testing.assert_close(v,state[k],rtol=0,atol=0)
   for k,v in crd2.state_dict().items(): torch.testing.assert_close(v,memories[k],rtol=0,atol=0)
if __name__=='__main__': unittest.main(verbosity=2)

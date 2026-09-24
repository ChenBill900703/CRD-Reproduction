from pathlib import Path
p=Path('reproduction/test_semantics.py'); s=p.read_text(); marker="if __name__=='__main__': unittest.main(verbosity=2)"; s=s.replace(marker,''' def test_epoch_boundary_loader_resume(self):
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
if __name__=='__main__': unittest.main(verbosity=2)''');p.write_text(s)

from pathlib import Path
p=Path('reproduction/test_preflight.py');s=p.read_text();marker="if __name__=='__main__':unittest.main(verbosity=2)"
new='''    def test_full_crd_upstream_parity_both_temperatures(self):
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
            self.assertLessEqual(max(allocated[2:])-min(allocated[2:]),1024*1024)
            self.assertTrue(all(torch.isfinite(v).all() for v in crd.state_dict().values()))
            assert_tree(self,teacher,t.state_dict());hs.remove();ht.remove()
            results.append(dict(tau=tau,steps=24,batch=64,K=16384,n_data=50000,seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),steady_allocated_growth_bytes=max(allocated[2:])-min(allocated[2:]),loss_first=losses[0],loss_last=losses[-1],data='synthetic; NOT benchmark accuracy'))
            del s,t,crd,opt,teacher;torch.cuda.empty_cache()
        (BASE/'artifacts/preflight/gpu-stress.json').write_text(json.dumps(results,indent=2))

'''
s=s.replace(marker,new+marker);p.write_text(s)

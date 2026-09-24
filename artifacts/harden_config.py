from pathlib import Path
p=Path('reproduction/reproduce.py'); s=p.read_text(); anchor='def main():\n'; insert='''def validate_config(c):
    fixed=dict(model_s='resnet8x4',model_t='resnet32x4',dataset='cifar100',test_batch_size=32,shuffle=True,drop_last=False,mode='exact',percent=1.0,feat_dim=128,n_data=50000,nce_m=.5,kd_T=4,learning_rate=.05,momentum=.9,weight_decay=.0005,lr_decay_epochs=[150,180,210],lr_decay_rate=.1,precision='fp32',tf32=False,cudnn_benchmark=False,cudnn_deterministic=True)
    for key,value in fixed.items():
        if c.get(key)!=value: raise ValueError(f'Unsupported setting {key}: {c.get(key)}')
    if c['loss_weights']!=list(WEIGHTS[c['method']]): raise ValueError('Loss weights differ')
    if c['nce_t'] != (.1 if c['profile']=='paper_tau' else .07): raise ValueError('Temperature profile differs')
    if c['profile']!='smoke' and (c['epochs']!=240 or c['batch_size']!=64 or c['nce_k']!=16384 or c.get('max_steps') or c.get('eval_steps')): raise ValueError('Formal config must be full benchmark')

'''; s=s.replace(anchor,insert+anchor); s=s.replace("c.update(seed=args.seed,trial=args.trial)","c.update(seed=args.seed,trial=args.trial); validate_config(c)"); s=s.replace("            ev=evaluate(s,test,device,c.get('eval_steps',0)); best=max(best,ev['top1'])","            train_seconds=time.perf_counter()-begin\n            ev=evaluate(s,test,device,c.get('eval_steps',0)); best=max(best,ev['top1'])"); s=s.replace("seconds=seconds,best_top1=best", "seconds=seconds,train_seconds=train_seconds,train_n=n,best_top1=best"); p.write_text(s)

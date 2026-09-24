from pathlib import Path
import json
for p in Path('configs').glob('*.json'):
 c=json.loads(p.read_text()); c.update(model_s='resnet8x4',model_t='resnet32x4',dataset='cifar100',test_batch_size=32,shuffle=True,drop_last=False,mode='exact',percent=1.0,feat_dim=128,n_data=50000,nce_m=.5,kd_T=4,learning_rate=.05,momentum=.9,weight_decay=.0005,lr_decay_epochs=[150,180,210],lr_decay_rate=.1,precision='fp32',tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,seed_list=[0,1,2,3,4],std_ddof=1)
 c['loss_weights']={'vanilla':[1,0,0],'kd':[.1,.9,0],'crd':[1,0,.8],'crd_kd':[1,1,.8]}[c['method']]
 p.write_text(json.dumps(c,indent=2))
# Existing official entrypoints keep their original code; --config opts into scoped reproducible runner.
for name,vanilla in [('train_teacher.py',True),('train_student.py',False)]:
 p=Path('reproduction')/name; s=p.read_text(); marker='import tensorboard_logger as tb_logger'
 insert='''# Reproduction config route: same supervised/distillation semantics, complete state logging.
import sys
if __name__ == '__main__' and '--config' in sys.argv:
    import json
    from pathlib import Path
    config_path = sys.argv[sys.argv.index('--config') + 1]
    method = json.loads(Path(config_path).read_text())['method']
    if (method == 'vanilla') != VANILLA:
        raise ValueError('Use train_teacher.py for vanilla and train_student.py for distillation')
    from reproduce import main
    main()
    raise SystemExit(0)

'''.replace('VANILLA',str(vanilla))
 s=s.replace(marker,insert+marker); p.write_text(s)

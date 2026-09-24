from pathlib import Path
import json
for p in Path('configs').glob('*.json'):
 c=json.loads(p.read_text());c.update(normalization_mean=[.5071,.4867,.4408],normalization_std=[.2675,.2565,.2761],augmentation=['RandomCrop(32,padding=4)','RandomHorizontalFlip()','ToTensor()','Normalize'],optimizer='SGD',nesterov=False,dampening=0,test_augmentation=['ToTensor()','Normalize'],checkpoint_boundary='epoch',num_gpus=1)
 p.write_text(json.dumps(c,indent=2))
p=Path('reproduction/reproduce.py');s=p.read_text().replace("teacher.load_state_dict(ck['model'],strict=True)","if ck.get('epoch') != 240: raise ValueError('Teacher is not epoch 240')\n        teacher.load_state_dict(ck['model'],strict=True)")
s=s.replace("precision='fp32',tf32=False,cudnn_benchmark=False,cudnn_deterministic=True)","precision='fp32',tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,normalization_mean=[.5071,.4867,.4408],normalization_std=[.2675,.2565,.2761],optimizer='SGD',nesterov=False,dampening=0,num_gpus=1)")
p.write_text(s)

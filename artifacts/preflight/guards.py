from pathlib import Path
p=Path('reproduction/reproduce.py');s=p.read_text().replace("    return (DataLoader(train", "    if len(train)!=50000 or len(test)!=10000: raise ValueError('Unexpected CIFAR-100 split sizes')\n    return (DataLoader(train")
s=s.replace("    if not all(torch.isfinite(grad).all().item() for grad in gradients):", "    if not torch.stack([torch.isfinite(grad).all() for grad in gradients]).all().item():")
s=s.replace("    opt.step()\n    acc=", "    opt.step()\n    if not torch.stack([torch.isfinite(p).all() for group in opt.param_groups for p in group['params']]).all().item(): raise FloatingPointError('Nonfinite parameter; discard step and resume last committed epoch')\n    acc=")
s=s.replace("status=dict(status='complete'", "status=dict(global_step=row['global_step'],train_n=row['train_n'],test_n=row['test_n'],code_fingerprint=hashlib.sha256(json.dumps(code_hashes(),sort_keys=True).encode()).hexdigest(),status='complete'")
p.write_text(s)
p=Path('reproduction/test_preflight.py');s=p.read_text().replace('from test_semantics import Semantics\n','').replace("            expected=test.dataset.transform.transforms[0](torch.zeros(1)) if False else None\n",'');p.write_text(s)

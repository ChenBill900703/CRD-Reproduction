from pathlib import Path
p=Path('reproduction/test_preflight.py');s=p.read_text().replace('import copy, importlib, io, json, shutil, subprocess, sys, tempfile, unittest','import copy, gc, importlib, io, json, shutil, subprocess, sys, tempfile, unittest')
s=s.replace('        for tau in (.07,.1):\n            c=config();','        for tau in (.07,.1):\n            gc.collect(); torch.cuda.empty_cache() # Isolate unreachable tensors from preceding test fixtures.\n            c=config();')
p.write_text(s)

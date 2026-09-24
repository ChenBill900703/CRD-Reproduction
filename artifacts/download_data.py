"""Download byte ranges from the official redirect target; enforce official MD5."""
import concurrent.futures, hashlib, json, urllib.request
from pathlib import Path
url='https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz'
size=169001437; n=16; root=Path('data/ranges'); root.mkdir(exist_ok=True)
def fetch_once(i):
 start=size*i//n; end=size*(i+1)//n-1
 if (root/str(i)).exists() and (root/str(i)).stat().st_size==end-start+1: return
 req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
 with urllib.request.urlopen(req,timeout=90) as r:
  if r.status!=206: raise RuntimeError('Server did not honor range')
  data=r.read()
 if len(data)!=end-start+1: raise RuntimeError('Incomplete range')
 (root/str(i)).write_bytes(data); print('range',i,'complete',flush=True)
def fetch(i):
 for attempt in range(5):
  try: return fetch_once(i)
  except Exception as e:
   print('retry',i,attempt,repr(e),flush=True)
   if attempt==4: raise
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(fetch,range(n)))
path=Path('data/cifar-100-verified.tar.gz')
with path.open('wb') as f:
 for i in range(n): f.write((root/str(i)).read_bytes())
assert hashlib.md5(path.read_bytes()).hexdigest()=='eb9058c3a382ffc7106e4002c42a8d85'
from torchvision.datasets.utils import extract_archive
extract_archive(str(path),'data')
Path('artifacts/data-provenance.json').write_text(json.dumps(dict(url=url,original_url='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',md5='eb9058c3a382ffc7106e4002c42a8d85',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=size),indent=2))
print('verified and extracted',flush=True)

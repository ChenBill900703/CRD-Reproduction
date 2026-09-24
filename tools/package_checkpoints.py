import json,zipfile,hashlib
from pathlib import Path
r=Path.cwd();out=r/'release-assets';out.mkdir(exist_ok=True)
q=json.loads((r/'artifacts/formal/queue.json').read_text())
p=out/'formal-checkpoints.zip'
with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
 for run in q['completed']:
  d=r/'runs'/run['name']
  for f in ['resume.pt','student_final.pt']:z.write(d/f,str((d/f).relative_to(r)))
 for f in ['reports/final-audit.json','artifacts/preflight/code-manifest.json']:z.write(r/f,f)
h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();(out/'SHA256SUMS.txt').write_text(h+'  '+p.name+'\n');print('Packaged',p.stat().st_size,'bytes SHA256',h)

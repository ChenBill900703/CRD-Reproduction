from pathlib import Path
import hashlib,difflib,json,subprocess
root=Path('.'); ref=root/'reference/RepDistiller'; work=root/'reproduction'
files=['README.md','scripts/run_cifar_distill.sh','scripts/run_cifar_vanilla.sh','scripts/fetch_pretrained_teachers.sh','train_student.py','train_teacher.py','helper/loops.py','helper/util.py','dataset/cifar100.py','crd/criterion.py','crd/memory.py','distiller_zoo/KD.py','models/resnet.py','models/__init__.py']
manifest={f:hashlib.sha256((ref/f).read_bytes()).hexdigest() for f in files}
Path('artifacts/reference-manifest.json').write_text(json.dumps(manifest,indent=2))
patch=[]
for f in files:
 patch.extend(difflib.unified_diff((ref/f).read_text().splitlines(True),(work/f).read_text().splitlines(True),fromfile='reference/'+f,tofile='reproduction/'+f))
Path('artifacts/compatibility.patch').write_text(''.join(patch))
with Path('artifacts/source-lines.txt').open('w',encoding='utf8') as out:
 for f in files:
  out.write('\n### '+f+'\n'); out.writelines(f'{i}: {line}\n' for i,line in enumerate((ref/f).read_text().splitlines(),1))

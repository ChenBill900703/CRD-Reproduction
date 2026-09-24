from pathlib import Path
import tarfile,hashlib,json
root=Path('data').resolve();archive=root/'cifar-100-verified.tar.gz'
assert hashlib.md5(archive.read_bytes()).hexdigest()=='eb9058c3a382ffc7106e4002c42a8d85'
expected={'train':'16019d7e3df5f24257cddd939b257f8d','test':'f0ef6b0ae62326f3e7ffdfab6717acfc','meta':'7973b15100ade9c7d40fb424638fde48'}
records=[]
with tarfile.open(archive,'r:gz') as tar:
 for member in tar.getmembers():
  if member.name not in ['cifar-100-python/'+n for n in expected]:continue
  assert member.isfile()
  target=root/member.name;assert target.resolve().is_relative_to(root)
  data=tar.extractfile(member).read();actual=hashlib.md5(data).hexdigest();assert actual==expected[target.name]
  target.parent.mkdir(exist_ok=True);target.write_bytes(data)
  records.append(dict(file=str(target),bytes=len(data),md5=actual))
assert len(records)==3
Path('artifacts/stage_b/extracted-data-verification.json').write_text(json.dumps(records,indent=2));print(records)

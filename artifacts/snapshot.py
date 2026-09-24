import ctypes,json,shutil,subprocess,sys
sys.path.insert(0,'reproduction')
from reproduce import environment,seed_all,sha
class Mem(ctypes.Structure):
 _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ['total','avail','total_page','avail_page','total_virtual','avail_virtual','extended']]
m=Mem();m.length=ctypes.sizeof(Mem);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));seed_all(0)
e=environment();e.update(ram_bytes=m.total,free_disk_bytes=shutil.disk_usage('.').free,driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version,memory.total','--format=csv,noheader'],text=True).strip())
from pathlib import Path
Path('artifacts/environment.json').write_text(json.dumps(e,indent=2));print(json.dumps(e,indent=2))

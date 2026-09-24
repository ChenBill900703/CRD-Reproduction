"""Final-epoch comparisons; failed/incomplete/inconsistent runs cannot be formal results."""
import csv,hashlib,json,math,statistics
from pathlib import Path
from reproduce import validate_config,TEACHER_SHA256
ROOT=Path(__file__).resolve().parent.parent
TARGET={'vanilla':72.50,'kd':73.33,'crd':75.51,'crd_kd':75.46}

def collect(root):
    rows=[]
    for folder in sorted(p for p in root.glob('*') if p.is_dir()):
        try:
            c=json.loads((folder/'config.json').read_text());s=json.loads((folder/'status.json').read_text())
        except (OSError,ValueError) as e:
            rows.append(dict(method=None,profile=None,seed=None,trial=None,epoch=None,final_top1=None,final_top5=None,best_top1=None,paper_target=None,gap_pp=None,status='unreadable',eligible=False,teacher_top1=None,teacher_sha256=None,seconds=None,peak_memory_bytes=None,config=str(folder/'config.json'),checkpoint=None,identity=None,error=str(e)))
            continue
        if c.get('evaluate_teacher'):continue
        reason=[]
        try:validate_config(c)
        except (ValueError,KeyError,TypeError) as e:reason.append(str(e))
        final=s.get('final_top1');teacher=s.get('teacher',{});method=c.get('method')
        valid_number=isinstance(final,(int,float)) and math.isfinite(final) and 0<=final<=100
        eligible=not reason and valid_number and s.get('status')=='complete' and s.get('epoch')==240 and c.get('epochs')==240 and c.get('profile') in ('author_code','paper_tau') and s.get('global_step')==782*240 and s.get('train_n')==50000 and s.get('test_n')==10000 and bool(s.get('code_fingerprint'))
        if method!='vanilla' and (teacher.get('sha256')!=TEACHER_SHA256 or teacher.get('n')!=10000):eligible=False;reason.append('Missing/mismatched teacher provenance')
        normalized={k:v for k,v in c.items() if k not in ('seed','trial','data','teacher','download')}
        try:env=json.loads((folder/'environment.json').read_text())
        except (OSError,ValueError):env=None;eligible=False;reason.append('Missing environment')
        identity=hashlib.sha256(json.dumps(dict(config=normalized,environment=env,code=s.get('code_fingerprint')),sort_keys=True).encode()).hexdigest()
        target=TARGET.get(method)
        rows.append(dict(method=method,profile=c.get('profile'),seed=c.get('seed'),trial=c.get('trial'),epoch=s.get('epoch'),final_top1=final,final_top5=s.get('final_top5'),best_top1=s.get('best_top1'),paper_target=target,gap_pp=final-target if eligible else None,status=s.get('status'),eligible=bool(eligible),teacher_top1=teacher.get('top1'),teacher_sha256=teacher.get('sha256'),seconds=s.get('elapsed_seconds'),peak_memory_bytes=s.get('peak_memory_bytes'),config=str(folder/'config.json'),checkpoint=s.get('checkpoint'),identity=identity,error='; '.join(reason)))
    return rows

def summarize(rows):
    out=[]
    for profile in ('author_code','paper_tau'):
        for method in TARGET:
            selected=[r for r in rows if r['eligible'] and r['profile']==profile and r['method']==method]
            if not selected:continue
            seeds=[r['seed'] for r in selected]
            consistent=len({r['identity'] for r in selected})==1 and len({r['teacher_sha256'] for r in selected})==1
            vals=[r['final_top1'] for r in selected]
            out.append(dict(profile=profile,method=method,seeds=seeds,n=len(vals),formal_five_seed=sorted(seeds)==[0,1,2,3,4] and consistent,consistent=consistent,mean=statistics.mean(vals) if consistent else None,std_sample_ddof1=statistics.stdev(vals) if consistent and len(vals)>1 else None,gap_pp=statistics.mean(vals)-TARGET[method] if consistent else None))
    return out

if __name__=='__main__':
    rows=collect(ROOT/'runs');out=ROOT/'artifacts';out.mkdir(exist_ok=True)
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summarize(rows),indent=2,allow_nan=False));print(json.dumps(summarize(rows),indent=2,allow_nan=False))

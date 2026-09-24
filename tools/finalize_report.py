import json, hashlib, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
expected=[(p,m,s) for p,ms in [('author_code',['vanilla','kd','crd','crd_kd']),('paper_tau',['crd','crd_kd'])] for s in range(5) for m in ms]
manifest=json.loads((R/'artifacts/preflight/code-manifest.json').read_text())
assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
rows=[]
for p,m,s in expected:
 d=R/'runs'/f'{p}_{m}_seed{s}_trial1'
 c=json.loads((d/'config.json').read_text());st=json.loads((d/'status.json').read_text())
 hist=[json.loads(l) for l in (d/'metrics.jsonl').read_text().splitlines()]
 assert st['status']=='complete' and st['epoch']==240 and st['global_step']==187680
 assert [v['epoch'] for v in hist]==list(range(1,241))
 assert c['seed']==s and c['profile']==p and c['method']==m
 assert hist[-1]['test_top1']==st['final_top1'] and hist[-1]['test_n']==10000
 if m!='vanilla':assert st['teacher']['sha256']=='22aa12e2b632b19ddde155ccf2388671842ddd304824b3095e0a7fff364beb4f'
 for name in ['resume.pt','student_final.pt','curves.png']:assert (d/name).is_file()
 rows.append(dict(profile=p,method=m,seed=s,final_top1=st['final_top1'],final_top5=st['final_top5'],best_top1=st['best_top1'],seconds=st['elapsed_seconds'],peak_memory_bytes=st['peak_memory_bytes'],run=str(d.relative_to(R))))
summary=json.loads((R/'artifacts/summary.json').read_text())
assert len(summary)==6 and all(x['formal_five_seed'] and x['consistent'] for x in summary)
(R/'reports/final-audit.json').write_text(json.dumps(dict(runs=30,epochs_per_run=240,seeds=[0,1,2,3,4],frozen_source_verified=True,teacher_checksum_verified=True,rows=rows),indent=2))
fig,ax=plt.subplots(figsize=(10,4.5))
labels=[x['profile']+'\n'+x['method'] for x in summary]
ax.bar(range(6),[x['mean'] for x in summary],yerr=[x['std_sample_ddof1'] for x in summary],capsize=5,color=['#78909c','#78909c','#247ba0','#247ba0','#70ad47','#70ad47'])
ax.scatter(range(6),[72.5,73.33,75.51,75.46,75.51,75.46],marker='D',color='#d1495b',label='Paper mean',zorder=3)
ax.set_xticks(range(6),labels);ax.set_ylim(71.5,76);ax.set_ylabel('CIFAR-100 final top-1 (%)');ax.set_title('Five seeds | epoch 240 | error bars: sample SD (ddof=1)');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(R/'reports/final-comparison.png',dpi=180);plt.close(fig)
text=['# 最終結果與查核報告','', '範圍：CIFAR-100，resnet32x4 → resnet8x4。30 runs 全數完成；這不是整篇論文的所有實驗。','', '所有主要比較取 epoch 240 final；std 使用樣本標準差 ddof=1。作者 seeds 與 std 分母未知。','', '| Profile | 方法 | mean ± std (%) | 論文差距（百分點） |','|---|---|---:|---:|']
for x in summary:text.append(f"| {x['profile']} | {x['method']} | {x['mean']:.3f} ± {x['std_sample_ddof1']:.3f} | {x['gap_pp']:+.3f} |")
text+=['','![比較圖](final-comparison.png)','','## 每次 final 結果','','| Profile | 方法 | Seed | Top-1 | Top-5 | Best（僅紀錄） | 訓練與評估時數 |','|---|---|---:|---:|---:|---:|---:|']
for r in rows:text.append(f"| {r['profile']} | {r['method']} | {r['seed']} | {r['final_top1']:.2f} | {r['final_top5']:.2f} | {r['best_top1']:.2f} | {r['seconds']/3600:.2f} |")
text+=['','## 解讀與限制','','- author_code 的 CRD 比 vanilla 高 2.840 個百分點，比 KD 高 2.154；主要蒸餾收益得到支持。','- author_code CRD+KD 比 CRD 僅高 0.002 個百分點，不能解讀為明確勝出。論文中略低也不是需要修正的錯誤。','- paper_tau 相對 author_code：CRD −0.120、CRD+KD −0.216 個百分點。只是一個模型配對、五個 seeds 的受控結果，不能推出普遍最佳 temperature。','- 兩種 temperature 分開報告，沒有挑較高者充當作者原始設定。','- 本機曾兩次非預期關機，另一次使用者要求重開機；CRD author_code seed3、paper_tau CRD+KD seed1/seed4 由完整 epoch checkpoint 接續。保留 events 與 console logs。','- resume 驗證涵蓋程式碼、config、環境、教師 checksum，恢復 optimizer、heads、memory/Z 與 RNG；不聲稱任意 batch 或跨硬體 bitwise 等價。','- seconds 只含已提交 epochs 的訓練及評估，不含停機、被丟棄的半個 epoch、存檔與啟動時間。','- 官方 test split 被原碼稱為 val；此處沒有另設 validation split，也未依 test 調參。','- ImageNet、跨模態、ensemble、表徵遷移及完整 benchmark 不在此次完成範圍。','', '查核明細：[final-audit.json](final-audit.json)。原始逐 epoch 曲線在各 runs/*/curves.png。']
(R/'reports/FINAL_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf8')
print('Verified 30 runs, 7200 epoch records, six five-seed groups; report generated.')

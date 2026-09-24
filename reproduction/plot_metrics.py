"""Plot recorded metrics. No interpolation or inferred benchmark results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent

def plot_runs(root):
    outputs=[]
    for p in root.glob('*/metrics.jsonl'):
        rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not rows:continue
        c=json.loads((p.parent/'config.json').read_text())
        x=[r['epoch'] for r in rows]
        fig,axes=plt.subplots(1,3,figsize=(13,3.5))
        try:
            for k in ['ce','kd','crd','total']:axes[0].plot(x,[r[k] for r in rows],marker='o',label=k)
            for k in ['train_top1','test_top1','test_top5']:axes[1].plot(x,[r[k] for r in rows],marker='o',label=k)
            axes[2].plot(x,[r['lr'] for r in rows],marker='o');axes[2].set_ylabel('Learning rate')
            label='Epoch (truncated smoke)' if c.get('max_steps') else 'Epoch'
            for a in axes:a.set_xlabel(label);a.grid(alpha=.2)
            axes[0].legend();axes[1].legend();axes[0].set_ylabel('Loss');axes[1].set_ylabel('Accuracy (%)')
            fig.suptitle(p.parent.name);fig.tight_layout();out=p.parent/'curves.png';fig.savefig(out,dpi=150);outputs.append(out)
        finally:plt.close(fig)
    return outputs
if __name__=='__main__':plot_runs(ROOT/'runs')

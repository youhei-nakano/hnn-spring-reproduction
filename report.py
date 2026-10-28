"""Summarize a fresh local official run; raw arrays/checkpoints stay in ignored runs/."""
import os,json,csv
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','./.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
R=Path(os.environ.get('HNN_REPORT_DIR',os.environ.get('HNN_RUN_DIR',P/'runs')))
def summarize(source):
    out={'training':source['training'],'evaluation':{}}
    for track in ['notebook_topdown_mutating','isolated_calls_diagnostic']:
        out['evaluation'][track]={k:{field:v[field] for field in ['energy_mse','error_bar']} for k,v in source['evaluation'][track].items()}
    out['evaluation']['evaluation_seconds']=source['evaluation']['evaluation_seconds']
    return out

def main():
    source=json.loads((R/'official_metrics.json').read_text());out=R/'summary';out.mkdir(exist_ok=True)
    (out/'official_metrics.json').write_text(json.dumps(summarize(source),indent=2))
    with (out/'table1_comparison.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['model','train_x1000','train_error_x1000','test_x1000','test_error_x1000','energy_x1000','energy_error_x1000'])
        for row in source['training']:
            e=source['evaluation']['notebook_topdown_mutating'][row['model']]
            w.writerow([row['model'],row['train_mse']*1000,row['train_error_bar']*1000,row['test_mse']*1000,row['test_error_bar']*1000,e['energy_mse']*1000,e['error_bar']*1000])
    fig,ax=plt.subplots(figsize=(7,3.5))
    for kind,col in [('nn','red'),('hnn','blue')]:
        a=np.loadtxt(R/f'official_{kind}_loss.csv',delimiter=',',skiprows=1);ax.semilogy(a[:,0],a[:,2],color=col,label=kind)
    ax.set(xlabel='update index',ylabel='test velocity MSE',title='Official spring protocol / seed 0');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'loss_curves.png',dpi=160);plt.close(fig)
    print('Aggregate-only output:',out)
if __name__=='__main__':main()

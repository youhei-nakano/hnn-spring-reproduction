# Independent audited port of Hamiltonian Neural Networks (2019) spring experiment.
# Upstream: Sam Greydanus, Misko Dzamba, Jason Yosinski.
# Reference commit: bcc362235dc623ffe48f22ccc22417e02e9803b4 (Apache-2.0).
# Changes: independent portable runner, explicit evaluation tracks and saved outputs.
"""著者コードを監査した上での独立移植。取得ソースは実行しない。"""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=os.environ.get('MPLCONFIGDIR','./.mplconfig'))
import time,json,hashlib,sys,shutil,csv
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent; R=Path(os.environ.get('HNN_RUN_DIR',P/'runs'));R.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1);torch.set_num_interop_threads(1)

def progress(s):
    print(s,flush=True);(P/'PROGRESS.md').write_text(s+'\n')
def dyn(t,z):
    q,p=np.split(z,2); return np.concatenate((p,-q))*2
def trajectory(radius=None,y0=None):
    t=np.linspace(0,3,30)
    if y0 is None:y0=np.random.rand(2)*2-1
    if radius is None:
        u=np.random.rand();radius=.1+.9*u
    z=y0/np.sqrt((y0*y0).sum())*radius
    sol=solve_ivp(lambda t,y:dyn(t,y),[0,3],z,t_eval=t,rtol=1e-10)
    assert sol.success
    clean=sol.y.T.copy();vel=np.stack([dyn(None,y) for y in clean])
    q,p=sol.y[0],sol.y[1];q+=np.random.randn(30)*.1;p+=np.random.randn(30)*.1
    return np.stack((q,p),-1),vel,clean,z

def dataset():
    np.random.seed(0); samples=[trajectory() for _ in range(50)]
    return {'x':np.concatenate([s[0] for s in samples]),'v':np.concatenate([s[1] for s in samples]),'clean':np.stack([s[2] for s in samples]),'initial':np.stack([s[3] for s in samples])}

class Model(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.a=nn.Linear(2,200);self.b=nn.Linear(200,200);self.c=nn.Linear(200,2,bias=False)
        for layer in [self.a,self.b,self.c]:nn.init.orthogonal_(layer.weight)
    def forward(self,x):return self.c(torch.tanh(self.b(torch.tanh(self.a(x)))))
    def velocity(self,x):
        if self.kind=='nn':return self(x)
        grad=torch.autograd.grad(self(x)[:,1].sum(),x,create_graph=True)[0]
        # J=[0,1;-1,0]、著者のdF2 @ M.Tと一致。
        return grad @ torch.tensor([[0.,-1.],[1.,0.]])

def train(track):
    d=dataset();np.savez_compressed(R/f'{track}_data.npz',**d)
    x=torch.tensor(d['x'][:750],dtype=torch.float32,requires_grad=True);v=torch.tensor(d['v'][:750],dtype=torch.float32)
    xt=torch.tensor(d['x'][750:],dtype=torch.float32,requires_grad=True);vt=torch.tensor(d['v'][750:],dtype=torch.float32)
    models={};rows=[];init=None
    for kind in ['nn','hnn']:
        assert shutil.disk_usage(P).free>10*1024**3
        torch.manual_seed(0);np.random.seed(0);m=Model(kind)
        state={k:t.clone() for k,t in m.state_dict().items()}
        if init is None:init=state
        else:assert all(torch.equal(t,init[k]) for k,t in state.items())
        torch.save(state,R/f'{track}_{kind}_initial.pt')
        opt=torch.optim.Adam(m.parameters(),lr=.001,weight_decay=.0001);history=[];start=time.perf_counter()
        updates=2001
        for step in range(updates):
            loss=(v-m.velocity(x)).square().mean();loss.backward();opt.step();opt.zero_grad()
            test=(vt-m.velocity(xt)).square().mean();history.append([step,loss.item(),test.item()])
            if step%500==0:progress(f'{track} {kind} update {step+1}/{updates}: {loss.item():.7g}')
        row={'model':kind,'updates':updates,'parameters':sum(t.numel() for t in m.parameters()),'training_seconds':time.perf_counter()-start}
        for label,xx,yy in [('train',x,v),('test',xt,vt)]:
            dist=(yy-m.velocity(xx)).square();row[label+'_mse']=dist.mean().item();row[label+'_error_bar']=dist.std().item()/np.sqrt(len(xx))
        np.savetxt(R/f'{track}_{kind}_loss.csv',history,delimiter=',',header='step,train_before_update,test_after_update',comments='')
        torch.save(m.state_dict(),R/f'{track}_{kind}.pt');models[kind]=m;rows.append(row)
    return models,rows

def model_solve(model,span,z,times):
    def fun(t,y):
        x=torch.tensor(y,dtype=torch.float32,requires_grad=True).view(1,2)
        return model.velocity(x).detach().numpy().reshape(-1)
    sol=solve_ivp(fun,span,z,t_eval=times,rtol=1e-12)
    if not sol.success:raise RuntimeError(sol.message)
    return sol.y.T,sol.nfev

def triplet(models,z,span,times):
    # 意図的なin-place変更。真軌道を先に評価する著者の順序を保つ。
    true_times=times.copy();true_span=span.copy()
    sol=solve_ivp(lambda t,y:dyn(t,y),span,z,t_eval=times,rtol=1e-12);assert sol.success
    span[1]*=1.09;times*=1.09
    b,bn=model_solve(models['nn'],span,z,times);h,hn=model_solve(models['hnn'],span,z,times)
    return {'truth':sol.y.T,'nn':b,'hnn':h,'true_t':true_times,'model_t':times.copy(),'true_span':true_span,'model_span':span.copy(),'nfev':[sol.nfev,bn,hn]}

def save_roll(path,obj):
    np.savez_compressed(path,**{k:np.asarray(v) for k,v in obj.items()})
def aggregate(losses):
    out={}
    for k,x in losses.items():
        a=np.array(x);group=a.reshape(5,3).mean(1)
        out[k]={'energy_mse':float(a.mean()),'error_bar':float(group.std(ddof=0)/np.sqrt(5)),'raw_losses':a.tolist(),'group_means':group.tolist()}
    return out

def evaluate(track,models):
    start=time.perf_counter()
    # notebook cell13 resets RNG via get_dataset; cell15 consumes 60 Gaussian values.
    dataset();trajectory(radius=1,y0=np.array([1.,0.]))
    rng_before=np.random.get_state();np.savez(R/f'{track}_eval_rng.npz',algorithm=rng_before[0],keys=rng_before[1],pos=rng_before[2],has_gauss=rng_before[3],cached_gaussian=rng_before[4])
    qualitative={}
    for k,m in models.items():
        t=np.linspace(0,30,2000);y,n=model_solve(m,[0,30],np.array([1.,0.]),t);qualitative[k]=y
        np.savez_compressed(R/f'{track}_qualitative_{k}.npz',t=t,y=y,nfev=n)
    span=[0,20];times=np.linspace(0,20,2000)
    illustrative=triplet(models,np.array([1.,0.]),span,times)
    save_roll(R/f'{track}_illustrative.npz',illustrative)
    times=np.linspace(span[0],span[1],100);losses={'nn':[],'hnn':[]};states=[];horizons=[]
    for i in range(15):
        z=np.random.rand(2)*1.6-.8;z+=.2*np.sign(z)*np.ones_like(z);states.append(z)
        obj=triplet(models,z,span,times);save_roll(R/f'{track}_mutating_{i:02}.npz',obj)
        for k in losses:losses[k].append(float(np.mean(((obj['truth']**2).sum(1)-(obj[k]**2).sum(1))**2)))
        horizons.append({'i':i,'true_end':obj['true_span'][1],'model_end':obj['model_span'][1]})
        progress(f'{track} notebook_topdown_mutating rollout {i+1}/15, model_end={span[1]:.5f}')
    main=aggregate(losses)
    # 同じ初期値を使う別診断。主トラックを上書きしない。
    isolated={'nn':[],'hnn':[]}
    for i,z in enumerate(states):
        obj=triplet(models,z,[0,20],np.linspace(0,20,100));save_roll(R/f'{track}_isolated_{i:02}.npz',obj)
        for k in isolated:isolated[k].append(float(np.mean(((obj['truth']**2).sum(1)-(obj[k]**2).sum(1))**2)))
    np.savez(R/f'{track}_eval_initial.npz',initial=np.array(states))
    result={'notebook_topdown_mutating':main,'isolated_calls_diagnostic':aggregate(isolated),'horizons':horizons,'evaluation_seconds':time.perf_counter()-start}
    # 保存図: 著者比較セルの軌道・座標誤差・学習H・真H。
    fig,ax=plt.subplots(1,4,figsize=(14,3.4))
    for k,color in [('truth','black'),('nn','red'),('hnn','blue')]:
        y=illustrative[k];ax[0].plot(y[:,0],y[:,1],color=color,label=k)
        if k!='truth':ax[1].plot(illustrative['model_t'],((y-illustrative['truth'])**2).mean(1),color=color,label=k)
        with torch.no_grad():hq=models['hnn'](torch.tensor(y,dtype=torch.float32))[:,1].numpy()
        ax[2].plot(illustrative['model_t'],hq,color=color);ax[3].plot(illustrative['model_t'],(y*y).sum(1),color=color)
    for a,title in zip(ax,['Predictions','Coordinate MSE','Learned H','True energy']):a.set_title(title);a.grid(alpha=.2)
    ax[0].set_xlabel('q');ax[0].set_ylabel('p');ax[0].legend();ax[1].set_xlabel('model time (1.09 scaled)')
    fig.tight_layout();fig.savefig(R/f'{track}_comparison.png',dpi=160);fig.savefig(R/f'{track}_comparison.svg');plt.close(fig)
    return result

def tests():
    x=torch.tensor([[.3,.4]],dtype=torch.float64,requires_grad=True);g=torch.autograd.grad((x*x).sum(),x)[0]
    assert np.array_equal(np.array([g[0,1].item(),-g[0,0].item()]),dyn(0,np.array([.3,.4])))
    sol=solve_ivp(lambda t,y:dyn(t,y),[0,3],[.3,.4],t_eval=np.linspace(0,3,30),rtol=1e-10)
    t=sol.t;exact=np.array([.3*np.cos(2*t)+.4*np.sin(2*t),.4*np.cos(2*t)-.3*np.sin(2*t)])
    assert abs(sol.y-exact).max()<3e-6
    (R/'tests.json').write_text(json.dumps({'hamiltonian_gradient':'passed','solve_ivp_reference_max_abs':float(abs(sol.y-exact).max()),'reference_only_not_data_generation':True},indent=2))

def main():
    (R/'environment.json').write_text(json.dumps({'python':sys.version,'executable':Path(sys.executable).name,'torch':torch.__version__,'scipy':scipy.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'threads':torch.get_num_threads()},indent=2))
    tests();all_results={};start=time.perf_counter()
    for track in ['official']:
        models,loss=train(track);result={'training':loss};(R/f'{track}_metrics.json').write_text(json.dumps(result,indent=2))
        result['evaluation']=evaluate(track,models);all_results[track]=result
        (R/f'{track}_metrics.json').write_text(json.dumps(result,indent=2))
        (R/'all_metrics.json').write_text(json.dumps(all_results,indent=2))
    progress(f'完了。公式コード規約×2モデル、著者solve_ivp評価、破壊的変更トラック＋独立呼出診断。総秒 {time.perf_counter()-start:.2f}')
if __name__=='__main__':main()

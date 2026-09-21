"""Small independent-cutoff PyMC models and posterior-mixture scores."""
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR',os.path.join(tempfile.gettempdir(),'ipo-tape-mpl'))
from pathlib import Path
from types import SimpleNamespace
import json, hashlib, warnings
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import xarray as xr
from scipy.special import logsumexp
from scipy import stats
from .tape_data import ROOT
MODEL_DIR=ROOT/'models';MODEL_DIR.mkdir(exist_ok=True)
SEED=20260908

def load_trace(path):
    """Load a shipped trace without requiring a NetCDF backend.

    Standalone releases store the groups used by the notebook in compressed
    NumPy archives. Development environments can still load the original
    NetCDF artifacts when h5netcdf or netCDF4 is installed.
    """
    path=Path(path)
    portable=path if path.suffix=='.npz' else path.with_suffix('.npz')
    if not portable.exists():
        return az.from_netcdf(path)
    groups={}
    with np.load(portable,allow_pickle=False) as archive:
        coefficient=archive['coefficient'].astype(str) if 'coefficient' in archive else np.array([],dtype=str)
        for group in ['posterior','prior','prior_predictive']:
            variables={}
            prefix=group+'__'
            for key in archive.files:
                if not key.startswith(prefix):continue
                name=key[len(prefix):]
                values=np.array(archive[key])
                if name=='beta':dims=('chain','draw','coefficient')
                elif name=='outcome':dims=('chain','draw','outcome_dim_0')
                else:dims=('chain','draw')
                variables[name]=(dims,values)
            if variables:
                coords={'coefficient':coefficient} if any('coefficient' in dims for dims,_ in variables.values()) else None
                groups[group]=xr.Dataset(variables,coords=coords)
    return SimpleNamespace(**groups)

def save_trace(path,idata,include_prior_predictive=True):
    """Persist only the posterior groups required for replay and prediction."""
    path=Path(path).with_suffix('.npz')
    arrays={}
    posterior=idata.posterior
    for name in ['alpha','beta','log_scale','delta_rv','nu']:
        if name in posterior:arrays['posterior__'+name]=np.asarray(posterior[name])
    if 'coefficient' in posterior.coords:arrays['coefficient']=np.asarray(posterior.coords['coefficient']).astype(str)
    if include_prior_predictive and hasattr(idata,'prior_predictive') and 'outcome' in idata.prior_predictive:
        arrays['prior_predictive__outcome']=np.asarray(idata.prior_predictive['outcome'])
    np.savez_compressed(path,**arrays)
    return path

def design_names(kind,t,size_conditioned=False):
    names=[]
    if kind not in ['M0','M0size'] and t>1:names+=['early_car']
    if kind in ['M2','M3','M2size']:names+=['log_ti']
    if kind in ['M0size','M1size','M2size']:names+=['log_size']
    if kind=='M1size' and t>1:names+=['price_size']
    if size_conditioned and 'log_size' not in names:names+=['log_size']
    return names

def matrix(df,names,means=None,scales=None):
    base=[n for n in names if n!='price_size']
    x=df[base].to_numpy(float) if base else np.empty((len(df),0))
    if means is None:means=x.mean(axis=0);scales=x.std(axis=0);scales=np.where(scales>1e-10,scales,1.)
    z=(x-means)/scales
    if 'price_size' in names:z=np.column_stack([z,z[:,base.index('early_car')]*z[:,base.index('log_size')]])
    return z,means,scales

def fit(train,kind,t,likelihood='student',tag='fit',prior_multiplier=1.,draws=800,tune=800,size_conditioned=False):
    names=design_names(kind,t,size_conditioned);x,means,scales=matrix(train,names)
    rv_mean=float(train.rv.mean());rv_sd=float(train.rv.std(ddof=0)) or 1.
    rv=(train.rv.to_numpy()-rv_mean)/rv_sd
    y=train.y.to_numpy(float)
    # Training-only robust scale: empirical Bayes, made explicit in notebook.
    ys=max(float((np.quantile(y,.75)-np.quantile(y,.25))/1.349),float(np.std(y))*.1,1e-4)
    specification=dict(kind=kind,cutoff=t,likelihood=likelihood,names=names,means=means.tolist(),scales=scales.tolist(),rv_mean=rv_mean,rv_sd=rv_sd,ys=ys,prior_multiplier=prior_multiplier,draws=draws,tune=tune,version=5)
    if size_conditioned:specification['size_conditioned']=True
    fingerprint=hashlib.sha256((json.dumps(specification,sort_keys=True)+train[['ipo_id','y',*sorted(set(names)-{'price_size'}),'rv']].to_csv(index=False)).encode()).hexdigest()[:16]
    path=MODEL_DIR/f'{tag}_{kind}_t{t}_{likelihood}_{fingerprint}.npz'
    meta_path=path.with_suffix('.json')
    if path.exists() or path.with_suffix('.npz').exists():return load_trace(path),json.loads(meta_path.read_text()),path
    print('FIT',tag,kind,t,likelihood,'n=',len(train),flush=True)
    with pm.Model(coords={'coefficient':names} if names else None) as model:
        alpha=pm.Normal('alpha',0,ys*prior_multiplier)
        mu=alpha
        if names:
            beta=pm.Normal('beta',0,ys*.5*prior_multiplier,dims='coefficient');mu=alpha+pm.math.dot(x,beta)
        log_scale=pm.Normal('log_scale',np.log(ys),.5*prior_multiplier)
        sigma=pm.math.exp(log_scale)
        if kind=='M3' and t>1:
            delta=pm.Normal('delta_rv',0,.35*prior_multiplier);sigma=pm.math.exp(log_scale+delta*rv)
        if likelihood=='student':
            nu=pm.Deterministic('nu',2+pm.Exponential('nu_minus_two',1/10));pm.StudentT('outcome',nu=nu,mu=mu,sigma=sigma,observed=y)
        else:pm.Normal('outcome',mu=mu,sigma=sigma,observed=y)
        prior=pm.sample_prior_predictive(samples=500,random_seed=SEED)
        idata=pm.sample(draws=draws,tune=tune,chains=4,cores=1,random_seed=SEED,target_accept=.9,progressbar=False,compute_convergence_checks=True,idata_kwargs={'log_likelihood':False})
        idata.extend(prior)
    summary=az.summary(idata,var_names=['alpha','log_scale']+(['beta'] if names else [])+(['delta_rv'] if kind=='M3' and t>1 else [])+(['nu'] if likelihood=='student' else []))
    specification.update(n_train=len(train),max_outcome_date=str(train.outcome_date.max()),max_listing_date=str(train.ipo_date.max()),fingerprint=fingerprint,
        rhat_max=float(summary.r_hat.max()),ess_bulk_min=float(summary.ess_bulk.min()),ess_tail_min=float(summary.ess_tail.min()),divergences=int(idata.sample_stats.diverging.sum()),bfmi_min=float(az.bfmi(idata).min()),
        prior_q=np.quantile(prior.prior_predictive.outcome.values,[.005,.025,.5,.975,.995]).tolist())
    save_trace(path,idata);meta_path.write_text(json.dumps(specification,indent=2));return idata,specification,path

def distribution(idata,meta,data,prior=False):
    post=idata.prior if prior else idata.posterior
    def flat(k):return np.asarray(post[k]).reshape((-1,)+np.asarray(post[k]).shape[2:])
    x,_,_=matrix(data,meta['names'],np.array(meta['means']),np.array(meta['scales']))
    mu=np.repeat(flat('alpha')[:,None],len(data),axis=1)
    if meta['names']:mu+=flat('beta')@x.T
    sig=np.exp(flat('log_scale'))[:,None]*np.ones((1,len(data)))
    if 'delta_rv' in post:sig*=np.exp(flat('delta_rv')[:,None]*(data.rv.to_numpy()-meta['rv_mean'])[None,:]/meta['rv_sd'])
    nu=flat('nu')[:,None] if meta['likelihood']=='student' else np.full((len(mu),1),np.inf)
    return mu,sig,nu

def predict(idata,meta,data,seed=SEED):
    mu,sig,nu=distribution(idata,meta,data)
    rng=np.random.default_rng(seed)
    draw=mu+sig*(rng.standard_t(nu,size=mu.shape) if meta['likelihood']=='student' else rng.normal(size=mu.shape))
    standardized=(data.y.to_numpy()[None,:]-mu)/sig if 'y' in data else None
    cdf=lambda threshold: (stats.t.cdf((threshold-mu)/sig,nu) if meta['likelihood']=='student' else stats.norm.cdf((threshold-mu)/sig)).mean(axis=0)
    out=data[['ipo_id','ipo_date','ipo_proceeds']].copy() if 'ipo_id' in data else pd.DataFrame(index=data.index)
    out['p_positive']=1-cdf(0);out['p_loss20']=cdf(np.log(.8))
    for q in [.025,.1,.25,.5,.75,.9,.975]:out[f'q{q:g}']=np.quantile(draw,q,axis=0)
    if standardized is not None:
        logpdf=(stats.t.logpdf(standardized,nu) if meta['likelihood']=='student' else stats.norm.logpdf(standardized))-np.log(sig)
        out['y']=data.y.to_numpy();out['log_score']=logsumexp(logpdf,axis=0)-np.log(len(mu))
        # Exact empirical-ensemble CRPS with O(S log S), no S-by-S array.
        sorted_draw=np.sort(draw,axis=0);s=len(draw)
        out['crps']=np.abs(draw-data.y.to_numpy()).mean(axis=0)-np.sum((2*np.arange(1,s+1)-s-1)[:,None]*sorted_draw,axis=0)/s**2
        out['pit']=cdf(data.y.to_numpy()[None,:])
        for level,lo,hi in [(50,.25,.75),(80,.1,.9),(95,.025,.975)]:out[f'cover{level}']=(out.y>=out[f'q{lo:g}'])&(out.y<=out[f'q{hi:g}'])
    return out,draw

def diagnostics_ok(meta):return meta['rhat_max']<=1.01 and meta['ess_bulk_min']>=400 and meta['ess_tail_min']>=400 and meta['divergences']==0 and meta['bfmi_min']>.3

def coefficients(idata,meta):
    rows=[]
    for name in meta['names']:
        x=np.asarray(idata.posterior.beta.sel(coefficient=name)).ravel()
        rows.append(dict(coefficient=name,median=np.median(x),lo=np.quantile(x,.025),hi=np.quantile(x,.975),p_positive=np.mean(x>0),p_negative=np.mean(x<0)))
    if 'delta_rv' in idata.posterior:
        x=np.asarray(idata.posterior.delta_rv).ravel();rows.append(dict(coefficient='delta_rv',median=np.median(x),lo=np.quantile(x,.025),hi=np.quantile(x,.975),p_positive=np.mean(x>0),p_negative=np.mean(x<0)))
    return pd.DataFrame(rows)

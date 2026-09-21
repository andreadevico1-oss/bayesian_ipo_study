"""Frozen historical-population prediction for any 1 <= cutoff < 60."""
from dataclasses import dataclass,field
import json
import numpy as np
import pandas as pd
from .tape_data import ROOT,OUT,features,cutoff_data,target
from .tape_models import fit,predict,diagnostics_ok,load_trace

@dataclass
class FrozenPopulation:
    selection:dict
    registry:list
    audit:pd.DataFrame
    panel:pd.DataFrame
    bundles:dict=field(default_factory=dict)
    def get(self,t,reference=False):
        # Left-continuous checkpoint policy, fixed before target data are supplied.
        checkpoint=max(c for c in self.selection['cutoffs'] if c<=t)
        selected=self.selection['selected_by_cutoff'][str(checkpoint)]
        kind=('M0size' if selected=='M0size' else 'M0') if reference else selected
        if t==1 and kind=='M1':kind='M0'
        if t==1 and kind=='M3':kind='M2'
        key=(t,kind)
        if key in self.bundles:return self.bundles[key]
        found=[r for r in self.registry if r['phase']=='final' and r['cutoff']==t and r['kind']==kind]
        if found:
            m=found[-1];bundle=(load_trace(ROOT/m['path']),m)
        else:
            # New cutoffs use exactly the same historical archive and fixed model
            # policy. No target row, target future or later IPO enters this fit.
            d=cutoff_data(self.audit,self.panel,t,self.selection['benchmark'])
            freeze=pd.Timestamp(self.selection['final_start'])
            d=d[(d.ipo_date<freeze)&(d.outcome_date<freeze)]
            idata,m,_=fit(d,kind,t,self.selection['likelihood'],tag='frozen_arbitrary',size_conditioned=self.selection.get('size_conditioned',False))
            if not diagnostics_ok(m):raise RuntimeError('New-cutoff sampling diagnostics failed; distribution not released')
            bundle=(idata,m)
        if not diagnostics_ok(bundle[1]):raise RuntimeError('Population diagnostics failed')
        self.bundles[key]=bundle;return bundle

def load_frozen_population():
    selection=json.loads((ROOT/'models/tape_selection.json').read_text())
    registry=json.loads((ROOT/'models/tape_registry.json').read_text())
    audit=pd.read_csv(OUT/'tape_sample_audit.csv',parse_dates=['ipo_date','outcome_date'])
    panel=pd.read_parquet(OUT/'tape_event_panel.parquet')
    return FrozenPopulation(selection,registry,audit,panel)

def _summary(row,draw,t,kind):
    return dict(calibration_status='Research forecast: broad-cohort intervals undercovered; $1bn+ validation has only 18 IPOs',cutoff=t,model=kind,draws_log_excess_return=draw[:,0],p_positive=float(row.p_positive),p_loss20=float(row.p_loss20),
        median_log=float(row['q0.5']),interval50_log=(float(row['q0.25']),float(row['q0.75'])),
        interval80_log=(float(row['q0.1']),float(row['q0.9'])),interval95_log=(float(row['q0.025']),float(row['q0.975'])))

def build_new_ipo_prior(population,*,offer_price,ipo_proceeds,ipo_date,market_state=None,verified_shares_offered=None):
    """Reference distribution of first-close-to-day-60 excess return before trading.
    Market state/shares accepted and recorded but not fitted covariates in this
    deliberately compact specification. This is not offer-to-day-60 return.
    """
    if not np.isfinite([offer_price,ipo_proceeds]).all() or offer_price<=0 or ipo_proceeds<=0:raise ValueError('Positive offering metadata required')
    ipo_date=pd.Timestamp(ipo_date)
    if pd.isna(ipo_date):raise ValueError('A valid listing date is required')
    if ipo_date<pd.Timestamp(population.selection['final_start']):raise ValueError('Target predates frozen population')
    idata,m=population.get(1,reference=True)
    if pd.Timestamp(m['max_outcome_date'])>=ipo_date:raise ValueError('Training outcomes unavailable before target IPO')
    d=pd.DataFrame([dict(ipo_id='new_ipo',ipo_date=ipo_date,ipo_proceeds=ipo_proceeds,log_size=np.log(ipo_proceeds),early_car=0.,log_ti=0.,rv=0.)])
    out,draw=predict(idata,m,d)
    result=_summary(out.iloc[0],draw,0,m['kind']);result['metadata']=dict(offer_price=offer_price,ipo_proceeds=ipo_proceeds,ipo_date=str(ipo_date),market_state=market_state,verified_shares_offered=verified_shares_offered)
    return result

def predict_remaining_day60_return(population,*,offer_price,ipo_proceeds,ipo_date,prices,volume,benchmark_returns,t,vix=None,market_state=None):
    """Arrays are ordered regular-market sessions from day 1. Future suffixes
    are sliced before validation or transforms. No interpolation of missing days.
    Prices/volume must use mutually consistent split units; TI is invariant.
    Benchmark entry 0 is unused because no prior regular-market IPO close exists.
    """
    if not isinstance(t,(int,np.integer)) or not 1<=t<60:raise ValueError('Require integer 1 <= t < 60')
    ipo_date=pd.Timestamp(ipo_date)
    if pd.isna(ipo_date):raise ValueError('A valid listing date is required')
    if ipo_date<pd.Timestamp(population.selection['final_start']):raise ValueError('Target predates frozen population')
    if not np.isfinite([offer_price,ipo_proceeds]).all() or offer_price<=0 or ipo_proceeds<=0:raise ValueError('Positive offering metadata required')
    c=np.asarray(prices)[:t];v=np.asarray(volume)[:t];b=np.asarray(benchmark_returns)[:t]
    if any(len(x)!=t for x in [c,v,b]):raise ValueError('Insufficient prefix data')
    benchmark=population.selection['benchmark']
    tape=pd.DataFrame(dict(day=np.arange(1,t+1),Close=c,Volume=v,**{benchmark+'_return':b}))
    meta=dict(offer_price=offer_price,ipo_proceeds=ipo_proceeds)
    x=features(meta,tape,t,benchmark)
    d=pd.DataFrame([{**x,'ipo_id':'new_ipo','ipo_date':ipo_date,'ipo_proceeds':ipo_proceeds}])
    idata,m=population.get(t)
    if pd.Timestamp(m['max_outcome_date'])>=ipo_date:raise ValueError('Training outcome leakage')
    out,draw=predict(idata,m,d)
    return _summary(out.iloc[0],draw,t,m['kind'])

def update_ipo_distribution(population,**kwargs):
    return predict_remaining_day60_return(population,**kwargs)

def pseudo_live(population):
    a=population.audit
    eligible=a[a.usable & (a.ipo_date>=pd.Timestamp(population.selection['final_start']))].copy()
    chosen=[]
    for _,g in eligible.groupby(eligible.ipo_date.dt.year):
        # Stable size ranks, including the largest IPO; no return-based selection.
        g=g.sort_values(['ipo_proceeds','ipo_date','ipo_id'])
        chosen.extend(g.iloc[np.unique(np.round(np.array([.2,.6,1.])*(len(g)-1)).astype(int))].ipo_id.tolist())
    rows=[];draws={}
    for rid in chosen:
        r=a.set_index('ipo_id').loc[rid];p=population.panel[population.panel.ipo_id.eq(rid)].sort_values('day')
        common=dict(offer_price=r.offer_price,ipo_proceeds=r.ipo_proceeds,ipo_date=r.ipo_date)
        forecasts=[build_new_ipo_prior(population,**common)]
        for t in [1,5,10,20]:forecasts.append(update_ipo_distribution(population,**common,prices=p.Close,volume=p.Volume,benchmark_returns=p[population.selection['benchmark']+'_return'],t=t,vix=p.vix))
        # Future outcome is read only after all distributions are constructed.
        for result in forecasts:
            t=result['cutoff'];draws[f'{rid}_{t}']=result['draws_log_excess_return']
            row={k:v for k,v in result.items() if k not in ['draws_log_excess_return','metadata']}
            row.update(ipo_id=rid,ticker=r.ticker,ipo_date=str(r.ipo_date),ipo_proceeds=r.ipo_proceeds,realized=target(p,max(t,1),population.selection['benchmark']))
            for level in [50,80,95]:row[f'lo{level}'],row[f'hi{level}']=row.pop(f'interval{level}_log')
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'tape_pseudo_live.csv',index=False)
    np.savez_compressed(ROOT/'models/tape_pseudo_live_draws.npz',**draws)
    return pd.DataFrame(rows)

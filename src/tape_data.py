"""Offline tape construction. Original caches and notebooks are never written."""
from pathlib import Path
import json, hashlib, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'
END=pd.Timestamp('2026-09-07')
# These transaction-type corrections do not depend on returns.
DIRECT={'WORK','PLTR','ASAN','COIN','RBLX','ZIP','AMPL','WRBY','BGXX','CLCO'}
SHELL={'FSDC','FEXD','MEKA','BDCI'}
IDENTITY={'IMG','NUZE','WAI','DPU'}

def load_inputs():
    u=pd.read_parquet(OUT/'ipo_universe_clean.parquet')
    u=u[u.is_operating_company_candidate].copy()
    u['ticker']=u.ticker.fillna('MISSING_SYMBOL')
    u['ipo_id']=u.ticker+'_'+u.ipo_date.dt.strftime('%Y%m%d')
    f=pd.read_parquet(OUT/'market_factor_daily.parquet').sort_index().loc[:END]
    # Recompute per-index returns before outer joins: holiday rows in other factors
    # must not turn the next valid U.S. return into NaN. No interpolation.
    for ticker in ['^GSPC','^IXIC']:
        f[ticker+'_log_return']=np.log(f[ticker+'_close'].dropna()).diff()
    return u,f

def features(metadata, tape, t, benchmark='broad', horizon=60):
    """Only accesses prefix 1:t. Day 1 aftermarket CAR and RV equal zero.

    Close and Volume must be in consistent split units. IR is intentionally
    diagnostic only because original caches lack a complete split ledger.
    """
    if not isinstance(t,(int,np.integer)) or not 1<=t<horizon: raise ValueError('Require integer 1 <= t < horizon')
    p=tape.sort_values('day').loc[lambda x:x.day<=t].copy()
    if p.day.tolist()!=list(range(1,t+1)): raise ValueError('Missing/duplicate prefix session')
    proceeds=float(metadata['ipo_proceeds'])
    if not np.isfinite(proceeds) or proceeds<=0: raise ValueError('Positive gross IPO proceeds required')
    c=p.Close.to_numpy(float);v=p.Volume.to_numpy(float);b=p[f'{benchmark}_return'].to_numpy(float)
    if not (np.isfinite(c).all() and (c>0).all() and np.isfinite(v).all() and (v>=0).all() and np.isfinite(b[1:]).all()):raise ValueError('Invalid prefix prices/volume/benchmark')
    r=np.diff(np.log(c)); early=float(np.sum(r-b[1:]))
    return dict(early_car=early,ti=float(np.sum(c*v)/proceeds),log_ti=float(np.log1p(np.sum(c*v)/proceeds)),rv=float(np.sqrt(np.sum(r*r))),
                log_size=float(np.log(proceeds)),drawdown=float(np.min(c/np.maximum.accumulate(c)-1)),
                feature_date=pd.Timestamp(p.date.iloc[-1]) if 'date' in p else pd.NaT)

def target(tape,t,benchmark='broad',horizon=60):
    p=tape.sort_values('day').set_index('day').reindex(range(1,horizon+1))
    c=p.Close.to_numpy(float); b=p[f'{benchmark}_return'].to_numpy(float)
    if not np.isfinite(c[t-1:]).all() or not np.isfinite(b[t:]).all():return np.nan
    return float(np.log(c[-1]/c[t-1])-b[t:].sum())

def build_data():
    u,f=load_inputs();cal=f.index[f['^GSPC_close'].notna()]
    cov=pd.read_csv(OUT/'coverage_by_horizon.csv');cov=cov[cov.horizon==60].set_index('ipo_id')
    records=[];panels=[]
    for _,r in u.iterrows():
        rid=r.ipo_id;a=cov.loc[rid];dates=cal[cal>=r.ipo_date][:60]
        reasons=[]
        if r.ticker in DIRECT: reasons.append('direct_listing_no_comparable_proceeds')
        if r.ticker in SHELL: reasons.append('blank_check_company')
        if r.ticker in IDENTITY:reasons.append('unresolved_identity_or_alias_duplicate')
        # Current Nasdaq name is diagnostic only; never a fitted feature.
        offer=pd.to_numeric(str(r.nasdaq_ipo_price).strip(),errors='coerce')
        if not np.isfinite(offer): offer=r.ipo_price
        proceeds=r.deal_size_usd
        pop=not reasons
        if not np.isfinite(proceeds) or proceeds<=0:reasons.append('missing_positive_proceeds')
        if not np.isfinite(offer) or offer<=0:reasons.append('missing_positive_offer_price')
        mature=len(dates)==60
        if not mature:reasons.append('immature')
        path=ROOT/a.source_path
        p=pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=['Close','Volume'])
        p.index=pd.to_datetime(p.index).tz_localize(None)
        duplicates=p.index.duplicated().any()
        if duplicates:reasons.append('duplicate_price_dates')
        p=p[~p.index.duplicated()].sort_index().reindex(dates)
        c=pd.to_numeric(p.Close,errors='coerce');v=pd.to_numeric(p.Volume,errors='coerce')
        split_path=ROOT/'data/tape_cache'/(rid+'_splits.json')
        split_known=False;factor=1.
        if split_path.exists():
            events=json.loads(split_path.read_text())
            split_known=events.get('status')==200 and events.get('splits') is not None
            if split_known:
                for event in events['splits'].values():
                    event_date=pd.Timestamp(event['date'],unit='s',tz='UTC').tz_convert('America/New_York').tz_localize(None).normalize()
                    if event_date>r.ipo_date:
                        factor*=float(event['numerator'])/float(event['denominator'])
        # Restore a constant IPO-date share basis. Products and log price ratios
        # are invariant; future split events are unit corrections, never predictors.
        c=c*factor;v=v/factor
        prices=bool(len(p)==60 and np.isfinite(c).all() and (c>0).all())
        volumes=bool(len(p)==60 and np.isfinite(v).all() and (v>=0).all() and v.sum()>0)
        if mature and not prices:reasons.append('incomplete_prices')
        if mature and not volumes:reasons.append('incomplete_volume')
        # Retain Nasdaq recoveries for audit/endpoint sensitivity; their split-volume
        # convention has not been verified, so main TI comparisons use Yahoo.
        if str(a.provider).lower().startswith('nasdaq'):reasons.append('unverified_nasdaq_adjustment')
        b=f.reindex(dates)
        if len(p) and (not np.isfinite(b['^GSPC_log_return'].iloc[1:]).all() or not np.isfinite(b['^IXIC_log_return'].iloc[1:]).all()):reasons.append('missing_benchmark')
        z=dict(ipo_id=rid,ticker=r.ticker,company_name=r.company_name,ipo_date=r.ipo_date,ipo_proceeds=proceeds,offer_price=offer,
               population=pop,mature=mature,price_complete=prices,volume_complete=volumes,usable=not reasons,
               exclusion_reason=';'.join(reasons),provider=a.provider,source_path=a.source_path,
               split_metadata_available=split_known,listing_unit_factor=factor,initial_return_provisional=np.log(c.iloc[0]/offer) if split_known and len(c) and c.iloc[0]>0 and offer>0 else np.nan,
               outcome_date=dates[-1] if mature else pd.NaT,initial_return_unverified=np.log(c.iloc[0]/offer) if len(c) and c.iloc[0]>0 and offer>0 else np.nan)
        records.append(z)
        if len(p):
            q=pd.DataFrame(dict(ipo_id=rid,day=np.arange(1,len(p)+1),date=dates,Close=c.to_numpy(),Volume=v.to_numpy(),
                broad_return=b['^GSPC_log_return'].to_numpy(),growth_return=b['^IXIC_log_return'].to_numpy(),vix=b['^VIX_close'].to_numpy()))
            q['return']=np.log(q.Close).diff();q['daily_ti']=q.Close*q.Volume/proceeds if proceeds>0 else np.nan
            panels.append(q)
    audit=pd.DataFrame(records);panel=pd.concat(panels,ignore_index=True)
    assert audit.ipo_id.is_unique
    audit.to_csv(OUT/'tape_sample_audit.csv',index=False);panel.to_parquet(OUT/'tape_event_panel.parquet',index=False)
    return audit,panel

def cutoff_data(audit,panel,t,benchmark='broad'):
    rows=[]
    for _,a in audit[audit.usable].iterrows():
        p=panel[panel.ipo_id.eq(a.ipo_id)]
        rows.append({**a.to_dict(),**features(a,p,t,benchmark), 'y':target(p,t,benchmark),'cutoff':t,'benchmark':benchmark})
    return pd.DataFrame(rows).sort_values(['ipo_date','ipo_id']).reset_index(drop=True)

def proxy_validation():
    s=pd.read_parquet(OUT/'check2_structural_v1.parquet')
    s=s[s.ipo_shares_offered_extraction_confidence.isin(['HIGH','MEDIUM']) & s.ipo_shares_offered.gt(0)].copy()
    s['implied']=s.deal_size_usd/s.ipo_price;s['absolute_error']=(s.implied-s.ipo_shares_offered).abs()
    s['relative_error']=(s.implied-s.ipo_shares_offered)/s.ipo_shares_offered
    s['log_error']=np.log(s.implied/s.ipo_shares_offered)
    s['log_actual']=np.log(s.ipo_shares_offered)
    s.to_csv(OUT/'tape_offered_share_validation.csv',index=False)
    return s,dict(n=len(s),mape=float(s.relative_error.abs().median()),relative_quantiles=s.relative_error.quantile([0,.1,.5,.9,1]).to_dict(),spearman=float(spearmanr(s.implied,s.ipo_shares_offered).statistic),log_error_sd=float(s.log_error.std()),log_actual_sd=float(s.log_actual.std()))

if __name__=='__main__':
    a,p=build_data();s,stats=proxy_validation()
    print(a.exclusion_reason.value_counts().to_string());print('USABLE',a.usable.sum(),'LARGE',(a.usable & a.ipo_proceeds.ge(1e9)).sum());print(stats)
    for b in ['broad','growth']:
        for t in [1,5,10,20]:cutoff_data(a,p,t,b).to_parquet(OUT/f'tape_features_{b}_{t}.parquet',index=False)

def independent_initial_returns(audit):
    """IPOScoop first close is a descriptive validation source, never a core input."""
    path=ROOT/'data/tape_cache/iposcoop_rated_2000_2020.xls'
    x=pd.read_excel(path,header=None)
    x=x.rename(columns={0:'ipo_date',1:'scoop_company',2:'ticker',4:'scoop_offer',6:'scoop_first_close'})
    x['ipo_date']=pd.to_datetime(x.ipo_date,errors='coerce');x['ticker']=x.ticker.astype(str).str.strip()
    for c in ['scoop_offer','scoop_first_close']:x[c]=pd.to_numeric(x[c],errors='coerce')
    x=x[x.ipo_date.ge('2019-01-01') & x.scoop_offer.gt(0) & x.scoop_first_close.gt(0)]
    x=x.drop_duplicates(['ticker','ipo_date'])
    z=audit.merge(x[['ipo_date','ticker','scoop_company','scoop_offer','scoop_first_close']],on=['ticker','ipo_date'],how='inner',validate='one_to_one')
    z=z[z.population].copy();z['initial_return_verified']=np.log(z.scoop_first_close/z.scoop_offer)
    z['offer_match']=np.isclose(z.offer_price,z.scoop_offer,rtol=1e-5)
    z.to_csv(OUT/'tape_initial_return_validation.csv',index=False)
    # Do not silently accept mismatched offer definitions.
    return z

def load_prepared():
    """Replay the released snapshot from portable derived artifacts."""
    a_path=OUT/'tape_sample_audit.csv';p_path=OUT/'tape_event_panel.parquet'
    if not a_path.exists() or not p_path.exists():
        return build_data()
    return pd.read_csv(a_path,parse_dates=['ipo_date','outcome_date']),pd.read_parquet(p_path)

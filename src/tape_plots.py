"""Single-purpose matplotlib figures; dark visual style from read-only reference."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from .tape_data import ROOT,OUT
FIG=ROOT/'reports/tape_figures';FIG.mkdir(parents=True,exist_ok=True)
C=['#65b5ff','#f7b65a','#55d5b0','#d79af5']

def style():
    plt.style.use('dark_background');plt.rcParams.update({'figure.figsize':(10,5),'figure.facecolor':'#0a0a0a','axes.facecolor':'#0a0a0a','savefig.facecolor':'#0a0a0a','font.size':11,'axes.grid':True,'grid.alpha':.2,'axes.spines.top':False,'axes.spines.right':False})

def save(fig,name):
    fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=160,bbox_inches='tight');return fig

def coverage(a):
    pop=a[a.population & a.mature].copy();pop['band']=pd.cut(pop.ipo_proceeds,[0,25e6,100e6,5e8,1e9,np.inf],labels=['<$25m','$25–100m','$100–500m','$500m–1bn','$1bn+'],right=False).cat.add_categories(['Unknown']).fillna('Unknown')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,col in zip(axs,['band','year']):
        if col=='year':pop['year']=pop.ipo_date.dt.year
        g=pop.groupby(col,observed=True).usable.agg(['size','sum'])
        x=np.arange(len(g));ax.bar(x,g['size'],color='#555555',label='Mature candidates');ax.bar(x,g['sum'],color=C[0],label='Usable tape')
        ax.set_xticks(x,g.index,rotation=35);ax.set_ylabel('Independent IPO episodes');ax.set_title('Coverage by '+('gross proceeds' if col=='band' else 'listing year'))
    axs[0].legend(fontsize=9);return save(fig,'01_coverage')

def distribution_hist(values,title,xlabel,name):
    fig,ax=plt.subplots();ax.hist(np.asarray(values)[np.isfinite(values)],bins=40,color=C[0],alpha=.8);ax.axvline(0,color='white',lw=1);ax.set(xlabel=xlabel,ylabel='IPO episodes',title=title);return save(fig,name)

def scatter(d,x,title,name):
    fig,ax=plt.subplots();big=d.ipo_proceeds>=1e9
    ax.scatter(d.loc[~big,x],d.loc[~big,'y'],s=18,alpha=.35,color=C[0],label='Below $1bn')
    ax.scatter(d.loc[big,x],d.loc[big,'y'],s=40,color=C[1],label='$1bn+')
    ax.axhline(0,color='white',lw=1);ax.set(xlabel={'early_car':'Early aftermarket excess log return','log_ti':'log(1 + cumulative dollar volume / gross IPO proceeds)'}[x],ylabel='Remaining day-60 excess log return',title=title);ax.legend();return save(fig,name)

def proxy(s):
    fig,ax=plt.subplots();ax.scatter(s.ipo_shares_offered,s.implied,color=C[0]);lo=min(s.ipo_shares_offered.min(),s.implied.min());hi=max(s.ipo_shares_offered.max(),s.implied.max());ax.plot([lo,hi],[lo,hi],color=C[1],ls='--');ax.set(xscale='log',yscale='log',xlabel='SEC-extracted base offered shares',ylabel='Gross proceeds / offer price',title=f'Offered-share validation: {len(s)} selective SEC observations');return save(fig,'04_share_proxy')

def prior_check(idata,train):
    y=idata.prior_predictive.outcome.values.ravel();lo,hi=np.quantile(y,[.01,.99]);fig,ax=plt.subplots();ax.hist(y,bins=np.linspace(lo,hi,100),density=True,color=C[0],alpha=.5,label='Prior predictive (central 98% shown)');ax.hist(train.y,bins=40,density=True,histtype='step',color=C[1],label='Training outcomes');ax.set(xlabel='Remaining excess log return',ylabel='Density',title='Data-calibrated prior predictive plausibility');ax.legend();return save(fig,'07_prior_predictive')

def coefficient_density(idata):
    fig,ax=plt.subplots();
    if 'beta' in idata.posterior:
        for j,n in enumerate(idata.posterior.coefficient.values):
            x=idata.posterior.beta.sel(coefficient=n).values.ravel();grid=np.linspace(*np.quantile(x,[.001,.999]),200);ax.plot(grid,gaussian_kde(x)(grid),color=C[j%4],label=str(n))
    if 'delta_rv' in idata.posterior:
        x=idata.posterior.delta_rv.values.ravel();grid=np.linspace(*np.quantile(x,[.001,.999]),200);ax.plot(grid,gaussian_kde(x)(grid),color=C[3],label='RV → log scale')
    ax.axvline(0,color='white',lw=1);ax.set(xlabel='Coefficient per training-standard-deviation change',ylabel='Posterior density',title='Day-10 posterior coefficients: effects and uncertainty');ax.legend();return save(fig,'08_coefficient_density')

def scores(s,benchmark,likelihood):
    s=s[(s.phase=='development')&(s.benchmark==benchmark)&(s.likelihood==likelihood)&s.kind.isin(['M0','M1','M2','M3'])]
    fig,ax=plt.subplots()
    for j,k in enumerate(['M0','M1','M2','M3']):
        g=s[s.kind==k].sort_values('cutoff');ax.plot(g.cutoff,g.log_score,'o-',color=C[j],label=k)
    ax.set(xlabel='Information cutoff (trading day)',ylabel='Mean held-out log predictive density ↑',title='Chronological development: incremental predictive information');ax.legend();return save(fig,'09_model_scores')

def calibration(s,selection):
    s=s[s.phase=='final'];fig,axs=plt.subplots(1,2,figsize=(11,4))
    for j,t in enumerate([1,5,10,20]):
        kind=selection['selected_by_cutoff'][str(t)]
        if t==1 and kind=='M1':kind='M0'
        if t==1 and kind=='M3':kind='M2'
        r=s[(s.cutoff==t)&(s.kind==kind)].iloc[0]
        x=np.array([50,80,95])/100;y=np.array([r[f'coverage{l}'] for l in [50,80,95]])
        lo=np.array([r[f'coverage{l}_lo'] for l in [50,80,95]]);hi=np.array([r[f'coverage{l}_hi'] for l in [50,80,95]])
        axs[0].errorbar(x+(j-1.5)*.004,y,yerr=[y-lo,hi-y],fmt='o-',color=C[j],label=f'Day {t}: {kind}',capsize=2)
    axs[0].plot([.4,1],[.4,1],ls='--',color='white');axs[0].set(xlabel='Nominal interval probability',ylabel='Held-out coverage (95% binomial intervals)',title='Final-cohort calibration');axs[0].legend(fontsize=8)
    p=pd.read_csv(OUT/'tape_predictions.csv');p=p[(p.phase=='final')&(p.cutoff==10)&(p.kind==selection['selected_by_cutoff']['10'])]
    axs[1].hist(p.pit,bins=np.linspace(0,1,11),color=C[0]);axs[1].axhline(len(p)/10,color=C[1],ls='--');axs[1].set(xlabel='Predictive CDF at realized day-10 target',ylabel='IPO episodes',title='Day-10 probability integral transform');return save(fig,'10_final_calibration')

def sequential(d):
    first=d.loc[d.ipo_proceeds.ge(1e9),'ipo_id'].iloc[0];g=d[d.ipo_id==first].sort_values('cutoff');fig,axs=plt.subplots(1,2,figsize=(11,4))
    for level,alpha in [(95,.12),(80,.2),(50,.3)]:axs[0].fill_between(g.cutoff,g[f'lo{level}'],g[f'hi{level}'],color=C[0],alpha=alpha,label=f'{level}% predictive')
    axs[0].plot(g.cutoff,g.median_log,'o-',color=C[0]);axs[0].plot(g.cutoff,g.realized,'x--',color=C[1],label='Realized remaining return (revealed last)');axs[0].set(xlabel='Trading-day cutoff; 0 = reference prior',ylabel='Remaining excess log return',title=f'{g.ticker.iloc[0]}: sequential distributions');axs[0].legend(fontsize=8)
    axs[1].plot(g.cutoff,g.p_positive,'o-',label='P(remaining excess > 0)',color=C[0]);axs[1].plot(g.cutoff,g.p_loss20,'o-',label='P(relative wealth loss > 20%)',color=C[1]);axs[1].set(xlabel='Trading-day cutoff',ylabel='Posterior predictive probability',ylim=(0,1),title='Probability updates');axs[1].legend(fontsize=8);return save(fig,'11_sequential')

def pseudo_densities(d):
    draws=np.load(ROOT/'models/tape_pseudo_live_draws.npz');ids=d.ipo_id.unique();fig,axs=plt.subplots(2,3,figsize=(13,7))
    for ax,rid in zip(axs.flat,ids):
        g=d[d.ipo_id==rid]
        for j,t in enumerate([0,5,10,20]):
            x=draws[f'{rid}_{t}'];grid=np.linspace(*np.quantile(x,[.01,.99]),200);ax.plot(grid,gaussian_kde(x)(grid),color=C[j],label=f'Day {t}')
        ax.set(title=f'{g.ticker.iloc[0]} | ${g.ipo_proceeds.iloc[0]/1e6:,.0f}m',xlabel='Remaining excess log return',ylabel='Predictive density')
    axs.flat[0].legend(fontsize=8);return save(fig,'12_pseudo_live_densities')

def normalization(d,variable,name):
    fig,ax=plt.subplots()
    for j,gname in enumerate(['all','$1bn+']):
        g=d[(d.variable==variable)&(d['group']==gname)].sort_values('day');ax.fill_between(g.day,g.q25,g.q75,color=C[j],alpha=.12);ax.plot(g.day,g.q50,color=C[j],label=gname+' median; IQR shading')
    ax.set(xlabel='Trading day',ylabel=variable.replace('_',' '),title='Event-time '+variable.replace('_',' '));ax.legend()
    if 'ratio' in variable:ax.axhline(1,color='white',ls='--',lw=1);ax.set_yscale('log')
    return save(fig,name)

def bounds(d):
    g=d[(d.cutoff==10)&(d.scope=='endpoint')&(d.event=='positive')].sort_values('threshold');fig,ax=plt.subplots()
    for j,(_,r) in enumerate(g.iterrows()):
        ax.plot([r.lower_lo,r.upper_hi],[j,j],lw=11,color=C[0],alpha=.25,label='Outer 95% endpoint envelope' if j==0 else None)
        ax.plot([r.lower_median,r.upper_median],[j,j],lw=4,color=C[0],label='Identification interval: median endpoints' if j==0 else None);ax.plot(r.q_median,j,'o',color=C[1],label='Observed-class posterior median' if j==0 else None)
    ax.legend(loc='lower right',fontsize=8)
    ax.set_yticks(range(len(g)),['All candidate sizes' if t==0 else f'≥ ${t/1e9:g}bn' for t in g.threshold]);ax.set(xlim=(0,1),xlabel='Population probability of positive remaining excess return',title='Missing outcomes: identification region + endpoint uncertainty');return save(fig,'17_identification_bounds')

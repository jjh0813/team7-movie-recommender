"""One frozen candidate, fixed ablations, and paired uncertainty. No test tuning."""
from pathlib import Path
import json
import pickle
import time
import numpy as np
import pandas as pd
from addons.team_modules import ROOT
from addons.full_reference import history_scores
from addons.hybrid import normalize,rank,score_request,ORIGINAL
from run_added import cohort

def fuse(ids,pop,eligible,c,b,cfg):
    cv=np.isfinite(c);bv=np.isfinite(b);alpha=cfg['alpha']
    if cfg['fusion']=='rrf':
        def ranks(v):
            x=np.full(len(ids),np.nan);ix=rank(ids,v,pop,len(ids));x[ix]=61/(60+np.arange(1,len(ix)+1));return x
        c=ranks(c);b=ranks(b)
    p=normalize(pop,eligible);den=alpha*cv+(1-alpha)*bv
    out=np.divide(alpha*np.nan_to_num(c)+(1-alpha)*np.nan_to_num(b),den,out=p.copy(),where=den>0)
    out=(1-cfg['pop_weight'])*out+cfg['pop_weight']*p;out[~eligible]=np.nan
    return out

def main():
    out=ROOT/'results/additional';frozen=json.loads((out/'frozen_selection.json').read_text(encoding='utf-8'))
    with (ROOT/'added_data/team_models.pkl').open('rb') as f:bundle=pickle.load(f)
    middle=dict(bundle['config']);chosen=dict(middle)
    chosen.update({k:frozen['best'][k] for k in ['alpha','pop_weight','fusion']})
    # Persist candidate settings BEFORE opening hidden outcomes.
    (out/'candidate_config.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8')
    cf=bundle['enhanced_cf'];ids=cf.movies_;cb=bundle['enhanced_content'];genre=bundle['original_content']
    history=pd.read_csv(ROOT/'added_data/request_history.csv');target=pd.read_csv(ROOT/'added_data/hidden_outcomes.csv')
    allowed=pd.read_csv(ROOT/'added_data/evaluation_candidates.csv').movieId.to_numpy()
    audit=json.loads((ROOT/'added_data/audit.json').read_text(encoding='utf-8'))
    assert history.timestamp.max()<=audit['last_training_timestamp']<target.timestamp.min()
    users,cohort_audit=cohort(history,target,allowed)
    baseline_path=ROOT/'results/baseline_v1/added_per_user.csv'
    baseline=pd.read_csv(baseline_path if baseline_path.exists() else ROOT/'results/added_per_user.csv')
    fixed_original=baseline[(baseline['mode']=='original')&(baseline.method=='Hybrid')].copy()
    assert set(fixed_original.userId)=={u for u,_,_ in users}
    records=[];started=time.perf_counter()
    # All diagnostic comparisons are predeclared. They cannot choose a new candidate.
    for index,(uid,h,truth) in enumerate(users):
        eligible=~np.isin(ids,h.movieId)&np.isin(ids,allowed)
        c=normalize(history_scores(cf,h),eligible)
        def content_scores(model):
            result=model.score(h,candidate_ids=ids);values=np.full(len(ids),np.nan)
            if result.status=='ok':values[result.scores.movieId.map(cf.mpos_).to_numpy(dtype=int)]=result.scores.content_score
            return normalize(values,eligible)
        b=content_scores(cb);g=content_scores(genre);pop=normalize(cf.popularity_,eligible)
        methods={
            'Intermediate hybrid':fuse(ids,cf.popularity_,eligible,c,b,middle),
            'Recent-validation hybrid':fuse(ids,cf.popularity_,eligible,c,b,chosen),
            'Ablation: no tags':fuse(ids,cf.popularity_,eligible,c,g,chosen),
            'Ablation: no popularity prior':fuse(ids,cf.popularity_,eligible,c,b,dict(chosen,pop_weight=0)),
            'Content with tags':np.where(np.isfinite(b),b,pop),
            'Popularity':pop,
        }
        # Match the actual component-ranking convention in score_request().
        cf_component=np.full(len(ids),np.nan)
        cf_order=rank(ids,c,cf.popularity_,len(ids))
        cf_component[cf_order]=61/(60+np.arange(1,len(cf_order)+1)) if chosen['fusion']=='rrf' else c[cf_order]
        methods['CF only']=np.where(np.isfinite(cf_component),cf_component,pop)
        for name,values in methods.items():
            values[~eligible]=np.nan;pos=rank(ids,values,cf.popularity_)
            r,n=ORIGINAL.ranking_metrics(ids[pos],truth,10)
            records.append({'userId':uid,'method':name,'recall_at_10':r,'ndcg_at_10':n})
        if (index+1)%300==0:print('External evaluated',index+1,'/',len(users),flush=True)
    detail=pd.DataFrame(records)
    original=fixed_original[['userId','recall_at_10','ndcg_at_10']].copy();original['method']='Original team hybrid'
    detail=pd.concat([original,detail],ignore_index=True)
    # Ensure the cache-based intermediate genuinely reproduces the submitted ZIP.
    old=baseline[(baseline['mode']=='enhanced')&(baseline.method=='Hybrid')].set_index('userId').sort_index()
    new=detail[detail.method=='Intermediate hybrid'].set_index('userId').sort_index()
    np.testing.assert_allclose(old[['recall_at_10','ndcg_at_10']],new[['recall_at_10','ndcg_at_10']],atol=1e-12)
    summary=detail.groupby('method').agg(users=('userId','size'),recall_at_10=('recall_at_10','mean'),ndcg_at_10=('ndcg_at_10','mean')).reset_index()
    detail.to_csv(out/'external_per_user.csv',index=False);summary.to_csv(out/'external_summary.csv',index=False)
    paired={};rng=np.random.default_rng(2026)
    for metric in ['recall_at_10','ndcg_at_10']:
        pivot=detail.pivot(index='userId',columns='method',values=metric)
        delta=(pivot['Recent-validation hybrid']-pivot['Intermediate hybrid']).to_numpy()
        samples=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(axis=1)
        paired[metric]={'mean_difference':float(delta.mean()),'ci95_low':float(np.quantile(samples,.025)),
                        'ci95_high':float(np.quantile(samples,.975)),'bootstrap_replicates':2000}
    (out/'paired_bootstrap.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
    # Candidate remains the validation-selected version, independent of external rank.
    bundle['config']=chosen
    with (ROOT/'added_data/team_models_v2.pkl').open('wb') as f:pickle.dump(bundle,f)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=['Original team hybrid','Intermediate hybrid','Recent-validation hybrid','CF only','Popularity']
    chart=summary.set_index('method').loc[order]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,metric in zip(axes,['recall_at_10','ndcg_at_10']):
        chart[metric].plot.bar(ax=ax,color=['#94A3B8','#64748B','#D97706','#0F766E','#CBD5E1'],rot=22)
        ax.set_title(metric);ax.set_xlabel('')
    fig.suptitle('Same 1,274 external cases; candidate selected on 32M only')
    fig.tight_layout();fig.savefig(out/'improvement_comparison.png',dpi=170);plt.close(fig)
    (out/'external_run_audit.json').write_text(json.dumps({'users':len(users),'seconds':time.perf_counter()-started,
        'cohort':cohort_audit,'intermediate_reproduced_exactly':True,'candidate_chosen_using_external_scores':False,
        'baseline_zip':json.loads((ROOT/'baseline_zip.json').read_text())},indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True);print('Paired differences',paired,flush=True)

if __name__=='__main__':main()

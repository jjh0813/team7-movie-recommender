"""Choose fusion settings on recent 32M periods before opening external outcomes."""
from pathlib import Path
import argparse
import json
import pickle
import zipfile
import numpy as np
import pandas as pd
from addons.team_modules import ROOT,content_based
from addons.full_reference import FullReferenceCF,history_scores,read
from addons.tag_content import TagContentExtension
from addons.hybrid import normalize,rank
from run_added import cohort

def collect(raw,year,max_users):
    cutoff=int(pd.Timestamp(f'{year}-01-01',tz='UTC').timestamp())
    end=int(pd.Timestamp(f'{year}-07-01',tz='UTC').timestamp())
    print('RECENT VALIDATION FIT',year,flush=True)
    base,movies,ratings=FullReferenceCF().fit_archive(raw/'ml-32m.zip',cutoff=cutoff)
    config=json.loads((ROOT/'addons/selected_config.json').read_text())
    config['min_support']=max(1,round(config['support_rate']*len(base.users_)))
    cf=base.configured_copy(config)
    cb=TagContentExtension().fit(movies,ratings);del ratings
    with zipfile.ZipFile(raw/'ml-32m.zip') as z:
        tags=read(z,'tags.csv');cb.add_tags(tags[tags.timestamp<cutoff],1.0);del tags
        target=[]
        for ch in read(z,'ratings.csv',chunksize=1000000):target.append(ch[(ch.timestamp>=cutoff)&(ch.timestamp<end)])
        target=pd.concat(target,ignore_index=True)
        # Randomize the target pool before reading profiles, but filter eligibility
        # before the final deterministic evaluation sample.
        uids=set(target.userId.unique());hist=[]
        for ch in read(z,'ratings.csv',chunksize=1000000):hist.append(ch[(ch.timestamp<cutoff)&ch.userId.isin(uids)])
    history=pd.concat(hist,ignore_index=True).sort_values(['userId','timestamp','movieId'])
    users,audit=cohort(history,target,cf.movies_)
    eligible=len(users)
    if len(users)>max_users:
        ix=np.random.default_rng(2026+year).choice(len(users),max_users,replace=False);users=[users[i] for i in sorted(ix)]
    # Cache only request scores and truth. No Beliefs files are read here.
    cache=[]
    for uid,h,truth in users:
        eligible_items=~np.isin(cf.movies_,h.movieId)
        c=normalize(history_scores(cf,h),eligible_items)
        result=cb.score(h,candidate_ids=cf.movies_)
        b=np.full(len(cf.movies_),np.nan)
        if result.status=='ok':b[result.scores.movieId.map(cf.mpos_).to_numpy(dtype=int)]=result.scores.content_score
        b=normalize(b,eligible_items);p=normalize(cf.popularity_,eligible_items)
        def ranks(v):
            out=np.full(len(v),np.nan);ix=rank(cf.movies_,v,cf.popularity_,len(v));out[ix]=61/(60+np.arange(1,len(ix)+1));return out
        cache.append({'userId':uid,'truth':truth,'c':c,'b':b,'cr':ranks(c),'br':ranks(b),'p':p,'eligible':eligible_items})
    audit.update({'year':year,'source':'ML32M only','eligible_users':eligible,'evaluated_users':len(users),
                  'cutoff_utc':str(pd.to_datetime(cutoff,unit='s',utc=True)),
                  'target_end_utc':str(pd.to_datetime(end,unit='s',utc=True)),
                  'reference_users':len(cf.users_),'reference_ratings':cf.source_rows_,
                  'tag_max_before_cutoff':True,'cf_min_support':config['min_support']})
    return {'ids':cf.movies_,'pop':cf.popularity_,'users':cache,'audit':audit}

def summarize(cache,fusion,alpha,pop_weight):
    recalls=[];ndcgs=[]
    for r in cache['users']:
        c,b=(r['cr'],r['br']) if fusion=='rrf' else (r['c'],r['b'])
        cv=np.isfinite(c);bv=np.isfinite(b);den=alpha*cv+(1-alpha)*bv
        score=np.divide(alpha*np.nan_to_num(c)+(1-alpha)*np.nan_to_num(b),den,out=r['p'].copy(),where=den>0)
        score=(1-pop_weight)*score+pop_weight*r['p'];score[~r['eligible']]=np.nan
        ix=rank(cache['ids'],score,cache['pop'])
        hits=np.isin(cache['ids'][ix],list(r['truth'])).astype(float)
        recalls.append(hits.sum()/len(r['truth']))
        ideal=(1/np.log2(np.arange(min(10,len(r['truth'])))+2)).sum()
        ndcgs.append((hits/np.log2(np.arange(len(hits))+2)).sum()/ideal)
    return float(np.mean(recalls)),float(np.mean(ndcgs))

def main():
    a=argparse.ArgumentParser();a.add_argument('--raw',type=Path,required=True);a.add_argument('--users',type=int,default=600)
    args=a.parse_args();out=ROOT/'results/additional';out.mkdir(parents=True,exist_ok=True)
    private=ROOT/'experiments/cache';private.mkdir(exist_ok=True)
    caches=[]
    for year in [2022,2023]:
        path=private/f'validation_{year}.pkl'
        if path.exists():
            with path.open('rb') as f:cache=pickle.load(f)
            if cache['audit']['evaluated_users']!=min(args.users,cache['audit']['eligible_users']):raise ValueError('Cached validation size mismatch')
        else:
            cache=collect(args.raw,year,args.users)
            with path.open('wb') as f:pickle.dump(cache,f)
        caches.append(cache);print('Validation cache ready',cache['audit'],flush=True)
    grid=[]
    # Include the frozen intermediate (.75, .20, rrf) as a declared comparator.
    for fusion in ['sum','rrf']:
        for alpha in [.5,.75,.85,.9,.95,.99]:
            for weight in [0,.05,.1,.2]:
                scores=[summarize(c,fusion,alpha,weight) for c in caches]
                grid.append({'fusion':fusion,'alpha':alpha,'pop_weight':weight,
                             'recall_2022':scores[0][0],'ndcg_2022':scores[0][1],
                             'recall_2023':scores[1][0],'ndcg_2023':scores[1][1],
                             'mean_recall':float(np.mean([s[0] for s in scores])),
                             'mean_ndcg':float(np.mean([s[1] for s in scores]))})
    frame=pd.DataFrame(grid).sort_values(['mean_ndcg','mean_recall','alpha'],ascending=[False,False,False],kind='stable')
    frame.to_csv(out/'recent_validation_search.csv',index=False)
    best=frame.iloc[0].to_dict()
    previous=frame[(frame.fusion=='rrf')&(frame.alpha==.75)&(frame.pop_weight==.2)].iloc[0].to_dict()
    frozen={'selection_source':'32M recent temporal validation only',
            'validation_audits':[c['audit'] for c in caches],'searched_configs':len(frame),
            'objective':'equal-period mean NDCG@10, then Recall@10; no Beliefs selection',
            'best':best,'intermediate_validation':previous,
            'external_outcomes_read_by_this_script':False}
    (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2,default=int),encoding='utf-8')
    print('FROZEN BEFORE ADDITIONAL EXTERNAL EVALUATION',json.dumps(best),flush=True)

if __name__=='__main__':main()

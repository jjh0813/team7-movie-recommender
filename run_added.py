"""New entry point that USES unmodified team modules. No web component."""
from pathlib import Path
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from addons.team_modules import ROOT
from addons.hybrid import ORIGINAL,score_request,rank

def load_model():
    newer=ROOT/'added_data/team_models_v2.pkl'
    with (newer if newer.exists() else ROOT/'added_data/team_models.pkl').open('rb') as f:return pickle.load(f)

def cohort(history,outcomes,allowed):
    groups={int(uid):g for uid,g in history.groupby('userId')}
    users=[];counts={'no_history':0,'no_supported_history':0,'no_relevant':0,'seen_positives_excluded':0,'positive_outside_catalog':0}
    allowed=set(allowed)
    for uid,g in outcomes.groupby('userId'):
        h=groups.get(int(uid))
        if h is None:counts['no_history']+=1;continue
        if not h.movieId.isin(allowed).any():counts['no_supported_history']+=1;continue
        liked=set(g.loc[g.rating>=4,'movieId'])
        counts['positive_outside_catalog']+=len(liked-allowed)
        counts['seen_positives_excluded']+=len(liked&allowed&set(h.movieId))
        truth=(liked&allowed)-set(h.movieId)
        if not truth:counts['no_relevant']+=1;continue
        users.append((int(uid),h,truth))
    counts['eligible_users']=len(users)
    return users,counts

def evaluate(bundle):
    d=ROOT/'added_data';out=ROOT/'results/current';out.mkdir(parents=True,exist_ok=True)
    history=pd.read_csv(d/'request_history.csv');truth=pd.read_csv(d/'hidden_outcomes.csv')
    allowed=pd.read_csv(d/'evaluation_candidates.csv').movieId.to_numpy()
    audit=json.loads((d/'audit.json').read_text(encoding='utf-8'))
    cutoff=audit['last_training_timestamp']
    assert history.timestamp.max()<=cutoff<truth.timestamp.min()
    users,counters=cohort(history,truth,allowed)
    if not users:raise ValueError('No eligible external cases')
    print('All eligible external users:',len(users),flush=True)
    rows=[]
    for mode in ['original','enhanced']:
        cf=bundle['original_cf'] if mode=='original' else bundle['enhanced_cf']
        for uid,h,relevant in users:
            values=score_request(bundle,h,mode,allowed)
            for method,score in values.items():
                pos=rank(cf.movies_,score,cf.popularity_)
                # Use the actual original metric function, not a substitute.
                recall,ndcg=ORIGINAL.ranking_metrics(cf.movies_[pos],relevant,10)
                rows.append({'mode':mode,'method':method,'userId':uid,'recall_at_10':recall,'ndcg_at_10':ndcg})
        print('Finished',mode,flush=True)
    details=pd.DataFrame(rows)
    summary=details.groupby(['mode','method']).agg(users=('userId','size'),recall_at_10=('recall_at_10','mean'),ndcg_at_10=('ndcg_at_10','mean')).reset_index()
    details.to_csv(out/'added_per_user.csv',index=False);summary.to_csv(out/'added_summary.csv',index=False)
    (out/'added_evaluation_audit.json').write_text(json.dumps(counters,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,m in zip(axes,['recall_at_10','ndcg_at_10']):
        summary.pivot(index='method',columns='mode',values=m).plot.bar(ax=ax,rot=10);ax.set_title(m)
    fig.tight_layout();fig.savefig(out/'added_performance.png',dpi=160);plt.close(fig)
    print(summary.to_string(index=False),flush=True)

def recommend(bundle,ratings,mode):
    from movie_content.content_based import validate_ratings
    h=validate_ratings(pd.read_csv(ratings))
    if h.movieId.duplicated().any():raise ValueError('Use one rating per movie')
    if 'userId' in h and h.userId.nunique()>1:raise ValueError('Only one user per request')
    values=score_request(bundle,h,mode);cf=bundle['original_cf'] if mode=='original' else bundle['enhanced_cf']
    ix=rank(cf.movies_,values['Hybrid'],cf.popularity_)
    result=bundle['movies'].set_index('movieId').loc[cf.movies_[ix]].reset_index()
    result['hybrid_score']=values['Hybrid'][ix]
    assert not result.movieId.isin(h.movieId).any()
    print('Mode:',mode);print(result.to_string(index=False))
    result.to_csv(ROOT/f'results/recommendations_{mode}.csv',index=False)
    return result

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('action',nargs='?',default='recommend',choices=['recommend','evaluate'])
    parser.add_argument('--ratings',type=Path,default=ROOT/'example_ratings.csv')
    parser.add_argument('--mode',choices=['original','enhanced'],default='enhanced',help='Default: final enhanced model. Original mode is kept for comparison only.')
    a=parser.parse_args(argv);model=load_model()
    if a.action=='recommend':recommend(model,a.ratings,a.mode)
    else:evaluate(model)

if __name__=='__main__':main()

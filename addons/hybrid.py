"""Adapters around the actual original combination and optional extensions."""
import numpy as np
import pandas as pd
from .team_modules import load_combiner
from .full_reference import history_scores

ORIGINAL=load_combiner()

def rank(ids,score,pop,k=10):
    valid=np.flatnonzero(np.isfinite(score));order=np.lexsort((ids[valid],-pop[valid],-score[valid]))
    return valid[order[:k]]

def normalize(raw,mask):
    finite=np.isfinite(raw)&mask;out=np.full(len(raw),np.nan)
    if finite.any():
        a,b=raw[finite].min(),raw[finite].max();out[finite]=(raw[finite]-a)/(b-a) if b>a else 0
    return out

def score_request(bundle,history,mode='enhanced',allowed=None):
    """Original mode executes the teammate's original hybrid_recommendation()."""
    cf=bundle['original_cf'] if mode=='original' else bundle['enhanced_cf']
    cb=bundle['original_content'] if mode=='original' else bundle['enhanced_content']
    ids=cf.movies_;eligible=~np.isin(ids,history.movieId)
    if allowed is not None:eligible&=np.isin(ids,allowed)
    cs=history_scores(cf,history);cb_res=cb.score(history,candidate_ids=ids)
    bs=np.full(len(ids),np.nan)
    if cb_res.status=='ok':
        p=cb_res.scores.movieId.map(cf.mpos_).to_numpy(dtype=int)
        bs[p]=cb_res.scores.content_score.to_numpy()
    pop=normalize(cf.popularity_,eligible)
    c=normalize(cs,eligible);b=normalize(bs,eligible)
    if mode=='original':
        class CFRequest:
            def score_all(self,uid):return cs
        class CBRequest:
            def score(self,h,candidate_ids=None):return cb_res
        # The temporary ID identifies this request ONLY; it is not linked to a
        # reference training ID. Every computation uses the supplied history.
        request=history.copy();request['userId']=0
        combined=ORIGINAL.hybrid_recommendation(0,CFRequest(),CBRequest(),request,ids,alpha=.6,k=len(ids))
        hybrid=np.full(len(ids),np.nan)
        if not combined.empty:
            hybrid[combined.movieId.map(cf.mpos_).to_numpy(dtype=int)]=combined.hybrid_score.to_numpy()
        missing=eligible&~np.isfinite(hybrid);hybrid[missing]=pop[missing]
    else:
        cfg=bundle['config'];a=cfg['alpha'];cv=np.isfinite(c);bv=np.isfinite(b)
        def rrf(values):
            result=np.full(len(ids),np.nan);ix=rank(ids,values,cf.popularity_,len(ids))
            result[ix]=61/(60+np.arange(1,len(ix)+1));return result
        if cfg['fusion']=='rrf':c=rrf(c);b=rrf(b)
        denominator=a*cv+(1-a)*bv
        hybrid=np.divide(a*np.nan_to_num(c)+(1-a)*np.nan_to_num(b),denominator,out=pop.copy(),where=denominator>0)
        hybrid=(1-cfg['pop_weight'])*hybrid+cfg['pop_weight']*pop
    values={'CF':np.where(np.isfinite(c),c,pop),'Content':np.where(np.isfinite(b),b,pop),
            'Hybrid':hybrid,'Popularity':pop}
    for v in values.values():v[~eligible]=np.nan
    return values

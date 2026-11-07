"""ONLY an optional scalable fit adapter. Original CF scoring is inherited intact."""
import copy
import zipfile
import numpy as np
import pandas as pd
from scipy import sparse
from .team_modules import cf_module,content_based

def read(z,name,**kwargs):
    n=next(n for n in z.namelist() if n.endswith('/'+name) and not n.startswith('__MACOSX'))
    return pd.read_csv(z.open(n),**kwargs)

class FullReferenceCF(cf_module.ItemBasedCF):
    """Do not override _score, score_from_ratings or the original fit().

    fit_archive() is an additional training entry point for the full 32M source.
    The original dense fit() and all original source remain available unchanged.
    """
    def fit_archive(self,archive,cutoff=None):
        dtype={'userId':'int32','movieId':'int32','rating':'float32','timestamp':'int64'}
        with zipfile.ZipFile(archive) as z:
            movies=read(z,'movies.csv');counts=pd.Series(dtype='int64');uids=set();total=0;last=0
            for ch in read(z,'ratings.csv',chunksize=1000000,dtype=dtype):
                ch=ch[ch.rating.between(.5,5)]
                if cutoff is not None:ch=ch[ch.timestamp<cutoff]
                if ch.empty:continue
                counts=counts.add(ch.movieId.value_counts(),fill_value=0)
                uids.update(ch.userId.unique());total+=len(ch);last=max(last,int(ch.timestamp.max()))
            m=movies[movies.genres.notna()&(movies.genres!='(no genres listed)')].copy()
            m['count']=m.movieId.map(counts).fillna(0)
            m=m.sort_values(['count','movieId'],ascending=[False,True]).head(self.n_movies).sort_values('movieId')
            self.movies_=m.movieId.to_numpy();self.mpos_={int(v):i for i,v in enumerate(self.movies_)}
            self.users_=np.array(sorted(uids));self.uidx_={int(v):i for i,v in enumerate(self.users_)}
            parts=[]
            for ch in read(z,'ratings.csv',chunksize=1000000,dtype=dtype):
                ch=ch[ch.rating.between(.5,5)]
                if cutoff is not None:ch=ch[ch.timestamp<cutoff]
                parts.append(ch[ch.movieId.isin(self.movies_)][['userId','movieId','rating']])
        ratings=pd.concat(parts,ignore_index=True)
        rows=np.searchsorted(self.users_,ratings.userId).astype('int32')
        cols=ratings.movieId.map(self.mpos_).to_numpy(dtype='int32')
        r=sparse.csr_matrix((ratings.rating.to_numpy(),(rows,cols)),shape=(len(self.users_),len(self.movies_)))
        observed=np.diff(r.indptr)
        self.user_mean_=np.divide(np.asarray(r.sum(axis=1)).ravel(),observed,out=np.zeros(len(observed)),where=observed>0).astype('float32')
        centered=r.copy();centered.data-=np.repeat(self.user_mean_,observed)
        mask=r.copy();mask.data[:]=1
        norm=np.sqrt(np.asarray(centered.multiply(centered).sum(axis=0)).ravel());norm[norm==0]=1
        raw=np.zeros((len(self.movies_),len(self.movies_)),dtype='float32');co=raw.copy()
        for a in range(0,len(self.movies_),256):
            b=min(a+256,len(self.movies_))
            raw[a:b]=(centered[:,a:b].T@centered).toarray()/np.outer(norm[a:b],norm)
            co[a:b]=(mask[:,a:b].T@mask).toarray()
        self.unfiltered_sim_=np.maximum(raw,0);np.fill_diagonal(self.unfiltered_sim_,0)
        self.common_support_=co
        self.popularity_=counts.reindex(self.movies_).to_numpy(dtype='float32')
        self.catalog_coverage_=len(ratings)/total
        self.co_mean_=float(co.mean());self.co_median_=float(np.median(co[np.triu_indices(len(self.movies_),1)]))
        self.hist_={}  # External requests use the ORIGINAL score_from_ratings(), never linked IDs.
        self.source_rows_=total;self.last_timestamp_=last;self.catalog_rows_=len(ratings)
        self.apply_support()
        return self,m[['movieId','title','genres']].copy(),ratings

    def apply_support(self):
        self.sim_=self.unfiltered_sim_.copy();self.sim_[self.common_support_<self.min_support]=0
        self.kept_=float((self.sim_>0).mean())

    def configured_copy(self,settings):
        model=copy.copy(self)
        for name in ['min_support','k','lam','min_neighbors']:setattr(model,name,settings[name])
        model.apply_support()
        return model

def history_scores(cf,history):
    """Call the teammate's actual method; do not reimplement prediction."""
    known=history[history.movieId.isin(cf.mpos_)].drop_duplicates('movieId',keep='last')
    if known.empty:return np.full(len(cf.movies_),np.nan)
    positions=known.movieId.map(cf.mpos_).to_numpy(dtype=int)
    return cf.score_from_ratings(positions,known.rating.to_numpy(dtype='float32'))

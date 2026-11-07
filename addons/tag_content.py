"""Optional extension; the teammate's ContentRecommender is neither edited nor replaced."""
import numpy as np
import pandas as pd
from scipy import sparse
import unicodedata
from .team_modules import content_based

class TagContentExtension(content_based.ContentRecommender):
    def add_tags(self,tags,tag_weight=1.0):
        t=tags[tags.movieId.isin(self.movie_index)].dropna(subset=['tag']).copy()
        t['feature']=t.tag.map(lambda s:' '.join(unicodedata.normalize('NFKC',str(s)).casefold().split()))
        t=t[t.feature.str.len().between(2,50)]
        df=t.groupby('feature').movieId.nunique();vocab=df[df>=3].sort_values(ascending=False).head(4000).index
        ti={s:i for i,s in enumerate(vocab)}
        counts=t[t.feature.isin(vocab)].groupby(['movieId','feature']).size().reset_index(name='count')
        weights=(1+np.log(counts['count']))*(np.log((1+len(self.catalog))/(1+counts.feature.map(df)))+1)
        mat=sparse.csr_matrix((weights,(counts.movieId.map(self.movie_index),counts.feature.map(ti))),shape=(len(self.catalog),len(ti)))
        norms=np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel());mat=sparse.diags(1/np.maximum(norms,1e-12))@mat
        features=sparse.hstack([sparse.csr_matrix(self.movie_vectors),tag_weight*mat],format='csr')
        norms=np.sqrt(np.asarray(features.multiply(features).sum(axis=1)).ravel())
        self.tag_vectors=sparse.diags(1/np.maximum(norms,1e-12))@features
        return self

    def score(self,history,candidate_ids=None):
        # Original input validation, candidate checks, seen exclusion, statuses and
        # genre explanation metadata all execute first, in the teammate's code.
        result=super().score(history,candidate_ids)
        if result.status!='ok' or not hasattr(self,'tag_vectors'):return result
        h=content_based.validate_ratings(history)
        if h.movieId.duplicated().any():h=h.sort_values('timestamp',kind='stable').drop_duplicates('movieId',keep='last')
        liked=h[(h.rating>=self.positive_threshold)&h.movieId.isin(self.movie_index)]
        pos=liked.movieId.map(self.movie_index).to_numpy()
        profile=np.asarray(self.tag_vectors[pos].T@liked.rating.to_numpy()).ravel()
        norm=np.linalg.norm(profile)
        if norm==0:return result
        ix=result.scores.movieId.map(self.movie_index).to_numpy()
        scores=np.asarray(self.tag_vectors[ix]@(profile/norm)).ravel()
        result.scores['content_score']=scores
        span=scores.max()-scores.min();result.has_ranking_signal=bool(span>1e-12)
        result.scores['content_score_normalized']=(scores-scores.min())/span if span>1e-12 else 0
        result.scores=result.scores.sort_values(['content_score','train_rating_count','movieId'],ascending=[False,False,True],kind='stable').reset_index(drop=True)
        return result

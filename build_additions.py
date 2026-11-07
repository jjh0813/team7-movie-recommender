"""Build extra full-reference artifacts WITHOUT changing any teammate source file."""
from pathlib import Path
import argparse
import json
import pickle
import zipfile
import numpy as np
import pandas as pd
from addons.team_modules import ROOT,content_based
from addons.full_reference import FullReferenceCF,read
from addons.tag_content import TagContentExtension

def clean_events(df):
    if 'tstamp' in df:
        df=df.rename(columns={'tstamp':'timestamp'})
        df.timestamp=pd.to_datetime(df.timestamp,utc=True,format='mixed',errors='coerce')
        df=df.dropna(subset=['timestamp']).copy();df.timestamp=df.timestamp.dt.as_unit('s').astype('int64')
    return df[df.rating.between(.5,5)].dropna(subset=['userId','movieId','timestamp']).sort_values(['userId','timestamp','movieId'])

def main():
    a=argparse.ArgumentParser();a.add_argument('--raw',type=Path,required=True);args=a.parse_args()
    dest=ROOT/'added_data';dest.mkdir(exist_ok=True)
    config=json.loads((ROOT/'addons/selected_config.json').read_text(encoding='utf-8'))
    print('Additional full-reference fit. Original team source files are not edited.',flush=True)
    base,movies,ratings=FullReferenceCF().fit_archive(args.raw/'ml-32m.zip')
    print('Original ContentRecommender.fit on all supported training observations:',len(ratings),flush=True)
    cb=content_based.ContentRecommender().fit(movies,ratings)
    del ratings
    enhanced=base.configured_copy(config)
    tag_cb=TagContentExtension();tag_cb.__dict__.update(cb.__dict__)
    with zipfile.ZipFile(args.raw/'ml-32m.zip') as z:tags=read(z,'tags.csv')
    tag_cb.add_tags(tags[tags.timestamp<=base.last_timestamp_],config.get('tag_weight',1))
    del tags
    # These matrices are fit-only intermediates. Inference still calls inherited
    # original _score() and score_from_ratings(); original dense fit() is untouched.
    for model in [base,enhanced]:
        del model.unfiltered_sim_;del model.common_support_
    model={'original_cf':base,'original_content':cb,'enhanced_cf':enhanced,
           'enhanced_content':tag_cb,'movies':movies,'config':config}
    with (dest/'team_models.pkl').open('wb') as f:pickle.dump(model,f)
    with zipfile.ZipFile(args.raw/'ml_belief_2024_data_release_2.zip') as z:
        bm=read(z,'movies.csv');events=clean_events(read(z,'user_rating_history.csv'))
    cutoff=base.last_timestamp_
    history=events[events.timestamp<=cutoff].drop_duplicates(['userId','movieId'],keep='last')
    outcomes=events[events.timestamp>cutoff].drop_duplicates(['userId','movieId'],keep='last')
    merged=movies.merge(bm,on='movieId',suffixes=('_a','_b'))
    norm=lambda s:s.fillna('').str.normalize('NFKC').str.casefold().str.replace(r'\s+',' ',regex=True).str.strip()
    allowed=merged.loc[norm(merged.title_a)==norm(merged.title_b),'movieId']
    history.to_csv(dest/'request_history.csv',index=False);outcomes.to_csv(dest/'hidden_outcomes.csv',index=False)
    pd.DataFrame({'movieId':allowed}).to_csv(dest/'evaluation_candidates.csv',index=False)
    movies.to_csv(dest/'movies.csv',index=False)
    audit={'source_rows':base.source_rows_,'source_users':len(base.users_),'catalog_rows':base.catalog_rows_,
           'candidate_movies':len(movies),'last_training_timestamp':cutoff,
           'last_training_utc':str(pd.to_datetime(cutoff,unit='s',utc=True)),
           'request_history_rows':len(history),'hidden_outcome_rows':len(outcomes),
           'matched_movies':len(allowed),'user_ids_linked':False,
           'original_files_modified':0,'enhancement_config_source':'previous 32M-only validation search; frozen before this evaluation'}
    (dest/'audit.json').write_text(json.dumps(audit,indent=2,default=int),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()

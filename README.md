# Team 7 Movie Recommender

A Python movie recommender that combines item-based memory collaborative filtering and content-based genre and tag scoring into one Top-10 list.

## Setup and recommendation

Use Python 3.12. Install the dependencies and download the original GroupLens archives before constructing the model.

```sh
python -m pip install -r requirements.txt
python download_data.py
python build_additions.py --raw data/raw
```

The build uses the full reference dataset and can take several minutes and several GB of memory. The first build creates the intermediate model and local evaluation data. To reproduce the final model, run:

```sh
python main.py evaluate
python -c "from pathlib import Path; from shutil import copyfile; p=Path('results/baseline_v1'); p.mkdir(exist_ok=True); copyfile('results/current/added_per_user.csv',p/'added_per_user.csv')"
python -m experiments.external_comparison
python main.py
```

Run the baseline evaluation before generating the final model so it evaluates the intermediate bundle. The final comparison uses the frozen configuration committed under results/additional and writes the final model locally. Supply a CSV with movieId and rating columns to use your own preferences:

```sh
python main.py --ratings my_ratings.csv
```

## Experiment

The final grid search compared 48 fusion settings over two chronological MovieLens 32M validation periods. To repeat the selection step, run:

```sh
python -m experiments.recent_validation --raw data/raw --users 600
```

The selected configuration uses reciprocal rank fusion with 85% CF and 15% content inside the personalized score, plus a separate 10% popularity contribution to the final score. On 1,274 eligible MovieLens Beliefs users, NDCG@10 rose from 0.08103 to 0.08455. The CF-only baseline scored 0.08692. See [the result table](results/additional/external_summary.csv) and [paired intervals](results/additional/paired_bootstrap.json).

## Data and attribution

- [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) supplies the training ratings and tags.
- [MovieLens Beliefs](https://grouplens.org/datasets/movielens/ml_belief_2024/) supplies external evaluation ratings.
- Dataset terms and citations are in [sources](sources). The project downloads the original archives directly from GroupLens.
- The original component scorers and integration code are in memory_cf, movie_content and code result. The final training, request and validation adapters are in addons and experiments.

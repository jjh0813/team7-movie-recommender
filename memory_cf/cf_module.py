# =============================================================================
#  Item-Based Memory Collaborative Filtering
#  Team 7 Movie Recommender  |  Machine Learning Term Project  |  황찬웅
# =============================================================================
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. 시간순 분할
# -----------------------------------------------------------------------------
def chronological_split(ratings, train_frac=0.7, val_frac=0.1):
    """사용자별 이력을 시간순으로 train / val / test 로 나눈다.

    무작위 분할은 미래 평점으로 과거를 예측하는 구조가 되어 현실의 추천
    상황과 맞지 않으므로 사용하지 않는다.
    """
    r = ratings.sort_values(['userId', 'timestamp'], kind='mergesort').reset_index(drop=True)
    rank = r.groupby('userId').cumcount()                      # 유저 내 순번
    size = r.groupby('userId')['movieId'].transform('size')    # 유저의 총 평점 수
    frac = rank / size
    r['split'] = np.where(frac < train_frac, 'train',
                   np.where(frac < train_frac + val_frac, 'val', 'test'))
    return r


# -----------------------------------------------------------------------------
# 2. 모델
# -----------------------------------------------------------------------------
class ItemBasedCF:
    """아이템 기반 메모리 협업 필터링 (adjusted cosine similarity).

    Parameters
    ----------
    n_movies : int
        학습셋 평점 수 기준 상위 몇 편을 후보 카탈로그로 쓸지.
    min_support : int
        두 영화를 모두 평가한 사용자 수가 이 값 미만이면 유사도를 폐기한다.
        공통 평가자가 적은 쌍의 유사도는 우연일 가능성이 높기 때문.
    k : int
        점수 계산에 사용할 이웃(유사 영화) 개수.
    lam : float
        shrinkage 계수. 분모를 키워 근거가 약한 후보의 점수를 사용자 평균
        쪽으로 끌어당긴다. 데이터가 부족할 때 극단적 추정을 피한다는 점에서
        정규화(regularization)와 같은 발상.
    min_neighbors : int
        유효 이웃이 이 값 미만인 후보는 점수를 내지 않고 fallback 처리한다.
    """

    def __init__(self, n_movies=3000, min_support=80, k=80,
                 lam=1000.0, min_neighbors=5):
        self.n_movies      = n_movies
        self.min_support   = min_support
        self.k             = k
        self.lam           = float(lam)
        self.min_neighbors = min_neighbors

    # ---------------------------------------------------------------- fit ---
    def fit(self, train, candidates=None):
        """학습셋만으로 후보 카탈로그 · 유사도 · 통계를 구축한다.

        candidates : 팀 공통 후보 목록(movieId). 주어지면 그대로 사용하고,
                     없으면 학습셋 평점 수 상위 n_movies 편을 직접 선정한다.
                     하이브리드 결합 시 모든 모듈이 동일한 후보를 써야 하므로
                     통합 실행에서는 반드시 이 인자를 지정한다.
        """
        counts = train['movieId'].value_counts()
        if candidates is not None:
            self.movies_ = np.asarray(candidates)
        else:
            self.movies_ = np.asarray(counts.head(self.n_movies).index)
        self.mpos_       = {m: i for i, m in enumerate(self.movies_)}
        self.popularity_ = counts.reindex(self.movies_).fillna(0).values.astype('float32')
        self.catalog_coverage_ = float(
            train['movieId'].isin(self.movies_).mean())

        # (2) 피벗 전에 후보 영화로 먼저 거른다 (메모리 보호)
        #     먼저 피벗하면 학습셋의 모든 영화로 열이 생겨 행렬이 수십 배 커진다.
        tf = train[train['movieId'].isin(self.movies_)]

        matrix = (tf.pivot_table(index='userId', columns='movieId', values='rating')
                    .reindex(columns=self.movies_)
                    .astype('float32'))
        self.users_ = np.asarray(matrix.index)
        self.uidx_  = {u: i for i, u in enumerate(self.users_)}

        # (3) 사용자 평균으로 중심화 — 평점 성향(후함/박함) 보정
        self.user_mean_ = matrix.mean(axis=1).values.astype('float32')
        C = matrix.values - self.user_mean_[:, None]          # NaN 유지
        self.C_ = C

        # (4) adjusted cosine similarity
        #     중심화 이후의 0 은 "그 사용자 평균과 같음(중립)" 이므로 0 채우기가 타당.
        #     sklearn cosine_similarity 는 내부에서 float64 로 승격되어 메모리를
        #     두 배 쓰므로 numpy 행렬곱으로 직접 계산한다.
        Cz = np.nan_to_num(C).astype('float32')
        norms = np.sqrt((Cz ** 2).sum(axis=0))
        norms[norms == 0] = 1.0
        sim = (Cz.T @ Cz) / np.outer(norms, norms)

        # (5) 공통 평가자 수 — 관측 마스크의 행렬곱
        mask = (~np.isnan(C)).astype('float32')
        co = mask.T @ mask
        self.co_mean_   = float(co.mean())
        self.co_median_ = float(np.median(co[np.triu_indices(len(self.movies_), k=1)]))

        # (6) 신뢰할 수 없는 유사도 제거
        sim[co < self.min_support] = 0.0   # 공통 평가자 부족
        sim[sim < 0] = 0.0                 # 음(-)의 유사도
        np.fill_diagonal(sim, 0.0)         # 자기 자신
        self.sim_  = sim
        self.kept_ = float((sim > 0).mean())

        # (7) 사용자별 시간순 이력 (짧은 이력 실험 · 웹 데모용)
        tf2 = tf.sort_values(['userId', 'timestamp'], kind='mergesort')
        pos = tf2['movieId'].map(self.mpos_).values
        self.hist_ = {}
        for uid, g in tf2.groupby('userId', sort=False).indices.items():
            self.hist_[uid] = (pos[g], tf2['rating'].values[g].astype('float32'))
        return self

    # ------------------------------------------------------------- scoring ---
    def _score(self, c, seen, umean):
        """중심화된 프로필 c 로 후보 전체 점수를 계산. 사용 불가 후보는 NaN."""
        out = np.full(len(self.movies_), np.nan, dtype='float32')
        ns = int(seen.sum())
        if ns == 0:
            return out

        Ssub = self.sim_[:, seen]                 # (후보, 본 영화)
        cs   = c[seen]

        kk = min(self.k, ns)
        if kk < ns:                               # 행마다 상위 k 개만 남김
            idx  = np.argpartition(-Ssub, kk - 1, axis=1)[:, :kk]
            rows = np.arange(Ssub.shape[0])[:, None]
            Stop = np.zeros_like(Ssub)
            Stop[rows, idx] = Ssub[rows, idx]
        else:
            Stop = Ssub

        num = Stop @ cs
        den = Stop.sum(axis=1)
        nb  = (Stop > 0).sum(axis=1)

        sc = umean + num / (den + self.lam)
        sc[nb < self.min_neighbors] = np.nan      # 근거 부족
        sc[seen] = np.nan                         # 이미 본 영화
        return sc.astype('float32')

    def score_from_ratings(self, movie_pos, ratings):
        """임의의 평점 프로필로 점수 계산.
        짧은 이력 실험과 웹 데모(신규 사용자 입력)에 사용한다."""
        c = np.full(len(self.movies_), np.nan, dtype='float32')
        c[movie_pos] = ratings
        umean = float(np.mean(ratings))
        c = c - umean
        return self._score(c, ~np.isnan(c), umean)

    def score_all(self, user_id, last_n=None):
        """학습셋에 존재하는 사용자의 후보 전체 점수.
        last_n 을 주면 가장 최근 last_n 개 평점만 사용한다."""
        if user_id not in self.hist_:
            return np.full(len(self.movies_), np.nan, dtype='float32')
        pos, rat = self.hist_[user_id]
        if last_n is not None:
            pos, rat = pos[-last_n:], rat[-last_n:]
        return self.score_from_ratings(pos, rat)

    # ----------------------------------------------------------- recommend ---
    def popular_topn(self, seen_pos, n=10):
        """학습셋 인기순 fallback 목록 (이미 본 영화 제외)."""
        order = np.argsort(-self.popularity_)
        mask = np.ones(len(self.movies_), dtype=bool)
        mask[seen_pos] = False
        return order[mask[order]][:n]

    def recommend(self, user_id, n=10, last_n=None):
        """Top-N 추천. 반환 (movieId 배열, fallback 사용 여부)."""
        sc  = self.score_all(user_id, last_n=last_n)
        pos = self.hist_.get(user_id, (np.array([], dtype=int),))[0]
        if last_n is not None:
            pos = pos[-last_n:]
        if not np.isfinite(sc).any():
            return self.movies_[self.popular_topn(pos, n)], True
        sc = np.where(np.isnan(sc), -np.inf, sc)
        t = np.argpartition(-sc, n)[:n]
        t = t[np.argsort(-sc[t])]
        return self.movies_[t], False


# -----------------------------------------------------------------------------
# 3. 평가
# -----------------------------------------------------------------------------
_LOG2 = np.log2(np.arange(2, 102))


def evaluate(model, holdout, n=10, users=None, last_n=None,
             popularity_only=False, rel_threshold=4.0):
    """Recall@n / NDCG@n / fallback / coverage.

    정답(relevant) = holdout 에서 rel_threshold 이상을 준 영화 중 후보 카탈로그에
    포함된 것. 학습셋에서 이미 평가한 영화는 후보에서 제외된다.
    """
    groups = holdout.groupby('userId')
    keys = list(groups.groups.keys()) if users is None else \
           [u for u in users if u in groups.groups]

    rec_l, ndcg_l, fb, cnt = [], [], 0, 0
    covered = set()

    for uid in keys:
        if uid not in model.hist_:
            continue
        g = groups.get_group(uid)
        rel = {model.mpos_[m] for m in g.loc[g['rating'] >= rel_threshold, 'movieId']
               if m in model.mpos_}
        if not rel:
            continue

        pos = model.hist_[uid][0]
        if last_n is not None:
            pos = pos[-last_n:]

        if popularity_only:
            t, used_fb = model.popular_topn(pos, n), False
        else:
            sc = model.score_all(uid, last_n=last_n)
            if not np.isfinite(sc).any():
                t, used_fb = model.popular_topn(pos, n), True
            else:
                sc = np.where(np.isnan(sc), -np.inf, sc)
                t = np.argpartition(-sc, n)[:n]
                t = t[np.argsort(-sc[t])]
                used_fb = False

        hits = np.fromiter((1.0 if x in rel else 0.0 for x in t), dtype='float64', count=len(t))
        rec_l.append(hits.sum() / len(rel))
        idcg = (1.0 / _LOG2[:min(n, len(rel))]).sum()
        ndcg_l.append((hits / _LOG2[:len(t)]).sum() / idcg)
        covered.update(t.tolist())
        fb += used_fb
        cnt += 1

    if cnt == 0:
        return {'users': 0}
    return {'users': cnt,
            f'Recall@{n}': round(float(np.mean(rec_l)), 4),
            f'NDCG@{n}'  : round(float(np.mean(ndcg_l)), 4),
            'fallback'   : round(fb / cnt * 100, 2),
            'coverage'   : round(len(covered) / len(model.movies_) * 100, 2)}

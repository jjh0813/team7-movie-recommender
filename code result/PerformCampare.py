import numpy as np
import pandas as pd
from cf_module import ItemBasedCF
from content_based import ContentRecommender

# =============================================================================
# 1. 랭킹 평가 지표 유틸리티 (Recall@K, NDCG@K)
# =============================================================================
def ranking_metrics(ranked_ids, relevant_ids, k=10):
    """Top-K 추천 결과에 대한 Recall과 NDCG 계산"""
    top = list(ranked_ids)[:k]
    hits = np.array([mid in relevant_ids for mid in top], dtype=float)
    
    recall = hits.sum() / len(relevant_ids) if relevant_ids else 0.0
    dcg = (hits / np.log2(np.arange(len(hits)) + 2)).sum()
    ideal = (1 / np.log2(np.arange(min(k, len(relevant_ids))) + 2)).sum()
    ndcg = float(dcg / ideal if ideal else 0.0)
    
    return recall, ndcg

# =============================================================================
# 2. 모델 단독 성능 비교 평가 (Memory-CF vs Content-Based)
# =============================================================================
def evaluate_models(train_df, test_df, movies_df, candidates, k=10, max_users=200):
    print("=" * 65)
    print(">>> [1/3] 메모리 기반 CF vs 콘텐츠 기반 모델 개별 성능 비교 평가")
    print("=" * 65)
    
    # 2.1 모델 학습 (Fitting) - 팀 공통 파라미터 적용
    cf_model = ItemBasedCF(min_support=80, k=80, lam=1000, min_neighbors=5)
    cf_model.fit(train_df, candidates=candidates)
    
    cb_model = ContentRecommender(positive_threshold=4.0)
    cb_model.fit(movies_df, train_df)

    # 2.2 테스트 대상 유저 추출
    users = sorted(test_df["userId"].unique())[:max_users]
    
    cf_recalls, cf_ndcgs = [], []
    cb_recalls, cb_ndcgs = [], []
    
    for uid in users:
        # 정답(Relevant): 테스트셋에서 4점 이상인 영화 중 후보 카탈로그에 포함된 것
        heldout = test_df[test_df["userId"] == uid]
        relevant = set(heldout.loc[heldout["rating"] >= 4.0, "movieId"]) & set(candidates)
        if not relevant:
            continue
            
        history = train_df[train_df["userId"] == uid]
        if history.empty:
            continue
            
        # 이미 학습 때 본 영화 제외
        relevant = relevant - set(history["movieId"])
        if not relevant:
            continue

        # --- A. Memory-CF 점수 산출 및 Top-K ---
        cf_sc = cf_model.score_all(uid)
        cf_df = pd.DataFrame({"movieId": candidates, "score": cf_sc})
        cf_df = cf_df.dropna(subset=["score"]).sort_values(by="score", ascending=False)
        cf_top_k = cf_df["movieId"].head(k).tolist()
        
        # --- B. Content-Based 점수 산출 및 Top-K ---
        cb_res = cb_model.score(history, candidate_ids=candidates)
        if cb_res.status == "ok":
            cb_top_k = cb_res.scores["movieId"].head(k).tolist()
        else:
            cb_top_k = []

        # --- C. 평가 지표 누적 ---
        cf_r, cf_n = ranking_metrics(cf_top_k, relevant, k)
        cb_r, cb_n = ranking_metrics(cb_top_k, relevant, k)
        
        cf_recalls.append(cf_r)
        cf_ndcgs.append(cf_n)
        cb_recalls.append(cb_r)
        cb_ndcgs.append(cb_n)

    print(f"[단독 모델 성능 비교 결과 (K={k})]")
    print(f" 1) Memory-Based CF : Recall@{k} = {np.mean(cf_recalls):.4f} | NDCG@{k} = {np.mean(cf_ndcgs):.4f}")
    print(f" 2) Content-Based   : Recall@{k} = {np.mean(cb_recalls):.4f} | NDCG@{k} = {np.mean(cb_ndcgs):.4f}")
    print("=" * 65 + "\n")
    
    return cf_model, cb_model

# =============================================================================
# 3. 통합(하이브리드) 추천 산출 및 가중치 결합
# =============================================================================
def hybrid_recommendation(uid, cf_model, cb_model, train_df, candidates, alpha=0.6, k=10):
    """
    Min-Max Scaling을 통해 각 모델의 점수 범위를 0~1로 맞춘 뒤 가중합 적용
    """
    history = train_df[train_df["userId"] == uid]
    seen_movies = set(history["movieId"])

    # 1. CF 점수 정규화 (Min-Max Scaling)
    cf_raw = cf_model.score_all(uid)
    cf_df = pd.DataFrame({"movieId": candidates, "cf_score": cf_raw})
    cf_df = cf_df.dropna(subset=["cf_score"])
    cf_df = cf_df[~cf_df["movieId"].isin(seen_movies)]
    
    if not cf_df.empty:
        c_min, c_max = cf_df["cf_score"].min(), cf_df["cf_score"].max()
        cf_df["cf_norm"] = (cf_df["cf_score"] - c_min) / (c_max - c_min) if c_max > c_min else 0.5
    else:
        cf_df["cf_norm"] = []

    # 2. CB 점수 정규화 (Min-Max Scaling)
    cb_res = cb_model.score(history, candidate_ids=candidates)
    if cb_res.status == "ok":
        cb_df = cb_res.scores[["movieId", "content_score"]].rename(columns={"content_score": "cb_score"})
        b_min, b_max = cb_df["cb_score"].min(), cb_df["cb_score"].max()
        cb_df["cb_norm"] = (cb_df["cb_score"] - b_min) / (b_max - b_min) if b_max > b_min else 0.5
    else:
        cb_df = pd.DataFrame(columns=["movieId", "cb_score", "cb_norm"])

    # 3. Outer Merge로 결합 (결측치는 0.0 처리)
    hybrid_df = pd.merge(cf_df[["movieId", "cf_norm"]], cb_df[["movieId", "cb_norm"]], on="movieId", how="outer")
    hybrid_df["cf_norm"] = hybrid_df["cf_norm"].fillna(0)
    hybrid_df["cb_norm"] = hybrid_df["cb_norm"].fillna(0)

    # 4. 콜드 스타트 대응 적응형 가중치(Adaptive Weight)
    # 평점 이력이 13개 미만인 경우 CF 성능이 떨어지므로 콘텐츠 기반(CB) 비중을 높임
    if len(history) < 13:
        w_cf, w_cb = 0.2, 0.8
    else:
        w_cf, w_cb = alpha, (1.0 - alpha)
        
    hybrid_df["hybrid_score"] = (w_cf * hybrid_df["cf_norm"]) + (w_cb * hybrid_df["cb_norm"])
    
    # 5. 최종 랭킹 정렬
    final_top_k = hybrid_df.sort_values(by="hybrid_score", ascending=False).head(k)
    return final_top_k

# =============================================================================
# 4. 메인 실행부
# =============================================================================
if __name__ == "__main__":
    print(">>> 팀 공통 데이터(shared/) 로드 중...")
    train = pd.read_csv("shared/train.csv")
    test = pd.read_csv("shared/test.csv")
    movies = pd.read_csv("shared/movies.csv")
    candidates = pd.read_csv("shared/candidate_ids.csv")["movieId"].values
    
    # 1. 단독 모델 성능 비교 평가 실행
    trained_cf, trained_cb = evaluate_models(train, test, movies, candidates, k=10, max_users=200)
    
    # 2. 하이브리드 추천 통합 결과 테스트 (첫 번째 테스트 유저 기준)
    test_uid = test["userId"].unique()[0]
    print("=" * 65)
    print(f">>> [2/3] [User {test_uid}] 하이브리드 추천 통합 결과 (Top-10)")
    print("=" * 65)
    hybrid_res = hybrid_recommendation(
        test_uid, trained_cf, trained_cb, train, candidates, alpha=0.6, k=10
    )
    print(hybrid_res.to_string(index=False))
    print("=" * 65)
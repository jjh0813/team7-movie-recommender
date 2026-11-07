
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


def integer_ids(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="raise")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite integer IDs")
    if (values < 0).any() or (values % 1 != 0).any():
        raise ValueError(f"{name} must contain nonnegative integer IDs")
    return values.astype("int64")


def validate_ratings(frame: pd.DataFrame, require_user: bool = False) -> pd.DataFrame:
    require_columns(frame, ["movieId", "rating"] + (["userId"] if require_user else []))
    result = frame.copy()
    result["movieId"] = integer_ids(result["movieId"], "movieId")
    if "userId" in result:
        result["userId"] = integer_ids(result["userId"], "userId")
    result["rating"] = pd.to_numeric(result["rating"], errors="raise")
    if not np.isfinite(result["rating"]).all() or not result["rating"].between(0.5, 5).all():
        raise ValueError("MovieLens ratings must be finite numbers from 0.5 to 5")
    if "timestamp" in result:
        result["timestamp"] = integer_ids(result["timestamp"], "timestamp")
    return result


def parse_genres(value) -> list[str]:
    
    if not isinstance(value, str):
        return []
    return sorted({g.strip() for g in value.split("|")
                   if g.strip() and g.strip() != "(no genres listed)"})


@dataclass
class ContentResult:
    scores: pd.DataFrame
    status: str
    reason: str
    profile: dict[str, float]
    positive_count: int
    ignored_history_movies: int
    has_ranking_signal: bool


class ContentRecommender:
    """4점 이상 평가한 영화로 사용자 취향을 계산한다."""

    def __init__(self, positive_threshold: float = 4.0):
        if not 0.5 <= positive_threshold <= 5:
            raise ValueError("positive_threshold must be between 0.5 and 5")
        self.positive_threshold = positive_threshold
        self._fitted = False

    def fit(self, movies: pd.DataFrame, train_ratings: pd.DataFrame):
        """장르 벡터와 학습 데이터의 영화별 평가 수를 준비한다."""
        require_columns(movies, ["movieId", "title", "genres"])
        catalog = movies[["movieId", "title", "genres"]].copy()
        catalog["movieId"] = integer_ids(catalog["movieId"], "movieId")
        if catalog["movieId"].duplicated().any():
            raise ValueError("Duplicate movie IDs in movies")
        if catalog["title"].isna().any():
            raise ValueError("Movie titles must not be missing")
        train = validate_ratings(train_ratings, require_user=True)
        if train.duplicated(["userId", "movieId"]).any():
            raise ValueError("Duplicate user/movie training rows: deduplicate before fit")
        counts = train.groupby("movieId").size()
        catalog["train_rating_count"] = catalog["movieId"].map(counts).fillna(0).astype(int)
        catalog["_genres"] = catalog["genres"].map(parse_genres)
        catalog = catalog[(catalog["train_rating_count"] > 0) & catalog["_genres"].map(bool)]
        catalog = catalog.sort_values("movieId").reset_index(drop=True)
        if catalog.empty:
            raise ValueError("No movies have both usable genres and training observations")
        self.genre_names = sorted({g for gs in catalog["_genres"] for g in gs})
        genre_index = {g: i for i, g in enumerate(self.genre_names)}
        matrix = np.zeros((len(catalog), len(self.genre_names)), dtype=np.float64)
        for row, gs in enumerate(catalog["_genres"]):
            matrix[row, [genre_index[g] for g in gs]] = 1.0
        # 영화 벡터의 길이를 1로 정규화
        self.movie_vectors = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        self.catalog = catalog
        self.movie_index = {int(mid): i for i, mid in enumerate(catalog["movieId"])}
        self._fitted = True
        return self

    def score(self, history: pd.DataFrame, candidate_ids: Iterable[int] | None = None) -> ContentResult:
        """한 사용자의 입력 평점으로 미평가 후보 전체의 점수를 계산한다."""
        if not self._fitted:
            raise RuntimeError("Call fit before score")
        history = validate_ratings(history)
        if "userId" in history and history["userId"].nunique() > 1:
            raise ValueError("Pass exactly one user's history; do not mix dataset users")
        if history["movieId"].duplicated().any():
            if "timestamp" not in history:
                raise ValueError("Repeated movie ratings require timestamp")
            tied = history.duplicated(["movieId", "timestamp"], keep=False)
            if (history[tied].groupby(["movieId", "timestamp"])["rating"].nunique() > 1).any():
                raise ValueError("Conflicting ratings at the same timestamp")
            history = history.sort_values("timestamp", kind="stable").drop_duplicates("movieId", keep="last")
        seen = set(history["movieId"])
        known = history["movieId"].isin(self.movie_index)
        ignored = int((~known).sum())
        likes = history[known & (history["rating"] >= self.positive_threshold)]
        if candidate_ids is None:
            requested = set(self.movie_index)
        else:
            requested = set(integer_ids(pd.Series(list(candidate_ids), dtype="object"), "candidate_ids"))
            unknown = requested - set(self.movie_index)
            if unknown:
                raise ValueError(f"Unsupported candidate IDs: {sorted(unknown)[:10]}")
        mask = self.catalog["movieId"].isin(requested - seen)
        scores = self.catalog.loc[mask, ["movieId", "title", "genres", "train_rating_count"]].copy()
        scores["content_score"] = np.nan
        scores["content_score_normalized"] = np.nan
        scores["shared_genres"] = ""
        if likes.empty:
            return ContentResult(scores.reset_index(drop=True), "no_profile",
                                 "No supported input movie rated at least the positive threshold",
                                 {}, 0, ignored, False)
        positions = [self.movie_index[int(mid)] for mid in likes["movieId"]]
        weights = likes["rating"].to_numpy(dtype=float)
        profile = np.average(self.movie_vectors[positions], axis=0, weights=weights)
        profile /= np.linalg.norm(profile)
        profile_map = dict(zip(self.genre_names, map(float, profile)))
        if scores.empty:
            return ContentResult(scores.reset_index(drop=True), "no_candidates", "No unseen candidates",
                                 profile_map, len(likes), ignored, False)
        # 정규화한 벡터의 내적으로 코사인 유사도 계산
        raw_scores = np.clip(self.movie_vectors[mask.to_numpy()] @ profile, 0.0, 1.0)
        scores["content_score"] = raw_scores
        span = float(raw_scores.max() - raw_scores.min())
        has_signal = span > 1e-12
        scores["content_score_normalized"] = ((raw_scores - raw_scores.min()) / span
                                               if has_signal else np.zeros(len(raw_scores)))
        active = {g for g, value in profile_map.items() if value > 0}
        scores["shared_genres"] = scores["genres"].map(
            lambda gs: "|".join(g for g in parse_genres(gs) if g in active))
        scores = scores.sort_values(
            ["content_score", "train_rating_count", "movieId"], ascending=[False, False, True],
            kind="stable").reset_index(drop=True)
        return ContentResult(scores, "ok", "", profile_map, len(likes), ignored, has_signal)

    def recommend(self, history: pd.DataFrame, k: int = 10,
                  candidate_ids: Iterable[int] | None = None) -> ContentResult:
        """추천 점수 상위 k개를 반환한다."""
        if k < 1:
            raise ValueError("k must be positive")
        result = self.score(history, candidate_ids)
        result.scores = result.scores.head(k).copy() if result.status == "ok" else result.scores.iloc[:0].copy()
        return result

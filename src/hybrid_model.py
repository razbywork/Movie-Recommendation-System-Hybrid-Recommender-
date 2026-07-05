import numpy as np
import pandas as pd


def normalize_scores(scores):
    scores = np.array(scores)

    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        return np.ones(len(scores))

    return (scores - min_score) / (max_score - min_score)


def hybrid_recommendations(
        als_recommendations,
        favorite_movie,
        content_df,
        similarity_matrix,
        alpha=0.7
):

    # ❗ use pandas index directly (safe mapping)
    movie_to_idx = pd.Series(
        content_df.index,
        index=content_df["title"]
    ).to_dict()

    if favorite_movie not in movie_to_idx:
        return []

    favorite_idx = movie_to_idx[favorite_movie]

    als_scores = [x["score"] for x in als_recommendations]
    normalized_als = normalize_scores(als_scores)

    results = []

    for rec, als_score in zip(als_recommendations, normalized_als):

        title = rec["title"]

        if title not in movie_to_idx:
            continue

        movie_idx = movie_to_idx[title]

        content_score = similarity_matrix[favorite_idx, movie_idx]

        final_score = (
            alpha * als_score
            + (1 - alpha) * content_score
        )

        results.append({
            "title": title,
            "als_score": round(float(als_score), 3),
            "content_score": round(float(content_score), 3),
            "hybrid_score": round(float(final_score), 3)
        })

    return sorted(
        results,
        key=lambda x: x["hybrid_score"],
        reverse=True
    )
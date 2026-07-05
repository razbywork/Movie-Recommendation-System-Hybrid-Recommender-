import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf_matrix(content_df):
    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(
        content_df["content"].fillna("")
    )

    return tfidf_matrix, tfidf


def compute_similarity(tfidf_matrix):
    return cosine_similarity(tfidf_matrix)


def recommend_content(movie_title, content_df, similarity_matrix, N=10):
    # Find movie index
    matches = content_df[content_df["title"] == movie_title]

    if matches.empty:
        return []

    idx = matches.index[0]

    # Similarity scores
    scores = list(enumerate(similarity_matrix[idx]))

    # Sort descending
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Skip the movie itself
    top_movies = scores[1:N + 1]

    results = []

    for i, score in top_movies:
        results.append({
            "movieId": content_df.iloc[i]["movieId"],
            "title": content_df.iloc[i]["title"],
            "score": round(float(score), 3)
        })

    return results


def build_user_profile(
        user_id,
        ratings,
        content_df,
        tfidf_matrix,
        min_rating=4.0
):
    liked_movies = ratings[
        (ratings["userId"] == user_id) &
        (ratings["rating"] >= min_rating)
    ]

    liked_movie_ids = liked_movies["movieId"].unique()

    indices = content_df[
        content_df["movieId"].isin(liked_movie_ids)
    ].index

    if len(indices) == 0:
        return None

    profile = np.asarray(
        tfidf_matrix[indices].mean(axis=0)
    )

    return profile


def recommend_from_profile(
        profile,
        content_df,
        tfidf_matrix,
        N=10
):
    similarities = cosine_similarity(
        profile,
        tfidf_matrix
    ).flatten()

    print("\nSimilarity Statistics")
    print("Max:", similarities.max())
    print("Min:", similarities.min())
    print("Mean:", similarities.mean())

    top_indices = similarities.argsort()[::-1][:N]

    results = []

    for idx in top_indices:
        results.append({
            "movieId": content_df.iloc[idx]["movieId"],
            "title": content_df.iloc[idx]["title"],
            "score": round(float(similarities[idx]), 3)
        })

    return results
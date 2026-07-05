import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


def create_user_item_matrix(ratings: pd.DataFrame):

    # convert ids to categorical indices
    user_ids = ratings["userId"].astype("category")
    movie_ids = ratings["movieId"].astype("category")

    # mappings
    user_index = user_ids.cat.codes
    movie_index = movie_ids.cat.codes

    user_map = dict(enumerate(user_ids.cat.categories))
    movie_map = dict(enumerate(movie_ids.cat.categories))

    user_inv_map = {v: k for k, v in user_map.items()}
    movie_inv_map = {v: k for k, v in movie_map.items()}

    # sparse matrix
    matrix = csr_matrix(
        (
            ratings["rating"].astype(float),
            (user_index, movie_index)
        )
    )

    return matrix, user_map, movie_map, user_inv_map, movie_inv_map

def train_als_model(
        user_item_matrix,
        factors=50,
        regularization=0.01,
        iterations=20
):

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations
    )

    # ALS expects item-user matrix
    model.fit(user_item_matrix)

    return model

def recommend_movies(
        model,
        user_id,
        user_item_matrix,
        movies_df,
        user_inv_map,
        movie_map,
        N=10
):

    user_idx = user_inv_map[user_id]

    movie_indices, scores = model.recommend(
        userid=user_idx,
        user_items=user_item_matrix[user_idx],
        N=N
    )

    results = []

    for movie_idx, score in zip(movie_indices, scores):

        movie_id = movie_map[movie_idx]

        movie_title = movies_df.loc[
            movies_df["movieId"] == movie_id,
            "title"
        ].values[0]

        results.append({
            "movieId": movie_id,
            "title": movie_title,
            "score": round(float(score), 3)
        })

    return results

def show_user_history(user_id, ratings, movies):
    user_data = ratings[ratings["userId"] == user_id]

    merged = user_data.merge(movies, on="movieId", how="left")

    return merged[["title", "rating"]].sort_values(by="rating", ascending=False)
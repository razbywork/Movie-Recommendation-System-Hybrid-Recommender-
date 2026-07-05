from src.data_loader import load_all
from src.preprocessing import (
    preprocess_movies,
    preprocess_tags,
    preprocess_ratings,
    merge_movies_tags
)

from src.collaborative_model import (
    create_user_item_matrix,
    train_als_model,
    recommend_movies,
    show_user_history
)

from src.content_model import (
    build_tfidf_matrix,
    compute_similarity,
    recommend_content,
    build_user_profile,
    recommend_from_profile
)

from src.hybrid_model import hybrid_recommendations


def main():
    print("Loading data...")
    ratings, movies, tags = load_all()

    print("Preprocessing...")

    ratings = preprocess_ratings(ratings)

    ratings = ratings.sample(1_000_000, random_state=42)

    valid_movie_ids = ratings["movieId"].unique()
    movies = movies[movies["movieId"].isin(valid_movie_ids)]
    tags = tags[tags["movieId"].isin(valid_movie_ids)]

    movies = preprocess_movies(movies)
    tags = preprocess_tags(tags)

    content_df = merge_movies_tags(movies, tags)

    print("Done!")

    print("\nCreating user-item matrix...")
    matrix, user_map, movie_map, user_inv_map, movie_inv_map = create_user_item_matrix(ratings)

    print("Training ALS model...")
    model = train_als_model(matrix)

    sample_user = ratings["userId"].value_counts().index[0]

    print("\n==============================")
    print(f"USER {sample_user} RECOMMENDATIONS")
    print("==============================\n")

    # ---------------- ALS ---------------- #
    als_recs = recommend_movies(
        model=model,
        user_id=sample_user,
        user_item_matrix=matrix,
        movies_df=movies,
        user_inv_map=user_inv_map,
        movie_map=movie_map,
        N=10
    )

    print("Collaborative Filtering (ALS):")
    for rec in als_recs:
        print(rec)

    print("\nUser History:")
    print(show_user_history(sample_user, ratings, movies).head(10))

    # ---------------- CONTENT ---------------- #
    print("\nContent-Based Model:")

    tfidf_matrix, _ = build_tfidf_matrix(content_df)
    similarity_matrix = compute_similarity(tfidf_matrix)

    sample_movie = "Forrest Gump (1994)"

    print(f"\nSimilar to: {sample_movie}\n")

    content_recs = recommend_content(
        sample_movie,
        content_df,
        similarity_matrix,
        N=10
    )

    for rec in content_recs:
        print(rec)

    # ---------------- HYBRID ---------------- #
    favorite_movie = show_user_history(sample_user, ratings, movies).iloc[0]["title"]

    print("\nHybrid Recommendations:\n")

    hybrid_recs = hybrid_recommendations(
        als_recommendations=als_recs,
        favorite_movie=favorite_movie,
        content_df=content_df,
        similarity_matrix=similarity_matrix,
        alpha=0.7
    )

    for rec in hybrid_recs:
        print(rec)

    print(f"\nBased on favorite movie: {favorite_movie}")

    # ---------------- USER PROFILE ---------------- #
    print("\nUser Profile Recommendations:\n")

    profile = build_user_profile(
        sample_user,
        ratings,
        content_df,
        tfidf_matrix
    )

    if profile is not None:
        profile_recs = recommend_from_profile(
            profile,
            content_df,
            tfidf_matrix,
            N=10
        )

        for rec in profile_recs:
            print(rec)

    print("\nSystem completed successfully.")

if __name__ == "__main__":
    main()
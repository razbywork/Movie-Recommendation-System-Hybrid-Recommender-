import pandas as pd


# ----------- Movies ----------- #
def preprocess_movies(movies: pd.DataFrame):
    movies = movies.copy()

    movies["genres"] = movies["genres"].fillna("")

    # clean "no genres listed"
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")

    movies["genres_list"] = movies["genres"].apply(
        lambda x: [g.strip() for g in x.split("|") if g.strip()]
    )

    return movies


# ----------- Tags ----------- #
def preprocess_tags(tags: pd.DataFrame):
    tags = tags.copy()

    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()

    tags = tags.drop_duplicates(subset=["userId", "movieId", "tag"])

    return tags


# ----------- Ratings ----------- #
def preprocess_ratings(ratings: pd.DataFrame):
    ratings = ratings.copy()

    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])

    return ratings


# ----------- Merge for Content ----------- #
def merge_movies_tags(movies, tags):
    tags_grouped = tags.groupby("movieId")["tag"].apply(
        lambda x: " ".join(x)
    ).reset_index()

    merged = movies.merge(tags_grouped, on="movieId", how="left")

    merged["tag"] = merged["tag"].fillna("").str.lower().str.strip()

    # clean genres text
    merged["genres_text"] = merged["genres"].fillna("").str.replace("|", " ")

    # final content feature
    merged["content"] = (merged["genres_text"] + " " + merged["tag"]).str.strip()

    return merged
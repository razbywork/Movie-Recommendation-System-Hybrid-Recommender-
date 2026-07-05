import pandas as pd
from pathlib import Path

def load_ratings():
    return pd.read_csv("ratings.csv")

def load_movies():
    return pd.read_csv("movies.csv")

def load_tags():
    return pd.read_csv("tags.csv")


def load_all():
    ratings = load_ratings()
    movies = load_movies()
    tags = load_tags()

    return ratings, movies, tags
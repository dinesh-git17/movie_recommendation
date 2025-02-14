# data_preprocessing.py
import pandas as pd
import os


def load_ratings(ratings_path="data/u.data"):
    """
    Loads ratings data from the MovieLens 100k dataset.
    Expected file is tab-separated with columns: userId, movieId, rating, timestamp.
    """
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(
            f"Could not find {ratings_path}. Please ensure the file is in the data folder."
        )
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
    )
    return ratings


def load_movies(movies_path="data/u.item"):
    """
    Loads movie data from the MovieLens 100k dataset.
    Expected file is pipe-separated with columns:
    movieId | title | release_date | video_release_date | IMDb_URL | [genre flags...]
    For simplicity, only movieId and title are extracted.
    """
    if not os.path.exists(movies_path):
        raise FileNotFoundError(
            f"Could not find {movies_path}. Please ensure the file is in the data folder."
        )
    # The u.item file has 24 columns; we'll assign names for the first few columns.
    movies = pd.read_csv(
        movies_path,
        sep="|",
        encoding="latin-1",
        header=None,
        names=[
            "movieId",
            "title",
            "release_date",
            "video_release_date",
            "IMDb_URL",
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ],
    )
    # Keep only movieId and title for our recommendation purposes.
    movies = movies[["movieId", "title"]]
    return movies


def merge_data():
    """
    Merges movies and ratings on movieId.
    """
    movies = load_movies()
    ratings = load_ratings()
    merged = pd.merge(ratings, movies, on="movieId")
    return merged


if __name__ == "__main__":
    data = merge_data()
    print("Merged data shape:", data.shape)

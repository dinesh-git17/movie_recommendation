import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from data_preprocessing import merge_data
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(
    "ignore", category=ConvergenceWarning, module="sklearn.decomposition._nmf"
)


def create_pivot_table(min_ratings=100):
    """
    Creates a pivot table (users x movies) with ratings.
    Only movies with at least min_ratings are retained.
    """
    data = merge_data()

    # Count number of ratings per movie
    ratings_count = data.groupby("title")["rating"].count()
    popular_movies = ratings_count[ratings_count >= min_ratings].index
    filtered_data = data[data["title"].isin(popular_movies)]

    # Create pivot table: rows = userId, columns = movie title, values = rating
    pivot = filtered_data.pivot_table(index="userId", columns="title", values="rating")
    return pivot


def advanced_recommendations(movie_title, pivot, n_components=20, top_n=10):
    """
    Generates recommendations using NMF-based matrix factorization.
    Fills missing ratings with 0, factorizes the matrix, and computes cosine similarities
    on the movie latent factors.
    """
    # Fill missing values with 0 (you might also experiment with other strategies)
    pivot_filled = pivot.fillna(0)

    # Apply NMF to factorize the matrix into user and movie latent factors
    nmf_model = NMF(n_components=n_components, init="random", random_state=42)
    W = nmf_model.fit_transform(pivot_filled)
    H = nmf_model.components_  # shape: (n_components, n_movies)

    # Transpose H to get movie latent factors: shape (n_movies, n_components)
    movie_factors = H.T
    movie_titles = pivot_filled.columns.tolist()

    # Compute cosine similarity between movies using the latent factors
    similarity_matrix = cosine_similarity(movie_factors)
    similarity_df = pd.DataFrame(
        similarity_matrix, index=movie_titles, columns=movie_titles
    )

    if movie_title not in similarity_df.index:
        raise ValueError(f"Movie '{movie_title}' not found in the dataset.")

    # Get the similarity series for the given movie and sort descending
    similar_movies = (
        similarity_df[movie_title]
        .drop(labels=[movie_title])
        .sort_values(ascending=False)
    )
    recommendations = similar_movies.head(top_n)
    return recommendations


if __name__ == "__main__":
    pivot = create_pivot_table()
    movie = "Toy Story (1995)"  # Adjust this title based on your data
    try:
        recs = advanced_recommendations(movie, pivot)
        print(f"Advanced Recommendations for '{movie}':")
        print(recs)
    except Exception as e:
        print(e)

from data_preprocessing import merge_data


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


def compute_similarity(pivot):
    """
    Computes the Pearson correlation matrix between movies.
    """
    # Use Pearson correlation and require a minimum number of common users
    correlation_matrix = pivot.corr(method="pearson", min_periods=100)
    return correlation_matrix


def get_recommendations(movie_title, pivot, correlation_matrix, top_n=10):
    """
    Returns top_n movie recommendations based on item correlation.
    """
    if movie_title not in correlation_matrix.columns:
        raise ValueError(f"Movie '{movie_title}' not found in the dataset.")

    # Get the correlation series for the given movie
    similar_movies = (
        correlation_matrix[movie_title].dropna().sort_values(ascending=False)
    )

    # Remove the movie itself and return the top_n
    recommendations = similar_movies.drop(labels=[movie_title]).head(top_n)
    return recommendations


if __name__ == "__main__":
    pivot = create_pivot_table()
    corr_matrix = compute_similarity(pivot)
    movie = "Toy Story (1995)"  # example movie; adjust as needed based on your data
    try:
        recs = get_recommendations(movie, pivot, corr_matrix)
        print(f"Recommendations for '{movie}':")
        print(recs)
    except Exception as e:
        print(e)

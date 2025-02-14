# dynamic_update.py
import pandas as pd
import os
import joblib
from data_preprocessing import merge_data, load_movies
from sklearn.decomposition import NMF


def update_dynamic_model(
    n_components=20, feedback_file="feedback.csv", output_model="dynamic_model.pkl"
):
    """
    Loads the original merged MovieLens data and appends user feedback as new ratings.
    Then, it creates an updated pivot table, trains an NMF model on the combined data,
    and saves the model and pivot table to a pickle file.

    Feedback CSV is expected to have columns:
    selected_movie, recommended_movie, similarity_score, user_rating, timestamp
    Each feedback entry is appended as a new rating from a virtual user (userId=999999).
    """
    # Load original merged data (columns: userId, movieId, rating, timestamp, title)
    merged = merge_data()

    # If feedback exists, load and incorporate it as additional ratings
    if os.path.exists(feedback_file):
        feedback = pd.read_csv(feedback_file)
        # Use a fixed virtual user id for feedback (could be changed or extended)
        virtual_user_id = 999999
        # Load movie metadata to map movie titles to movieIds
        movies_df = load_movies()  # columns: movieId, title
        # Merge feedback with movies data on recommended_movie == title
        feedback_merged = pd.merge(
            feedback,
            movies_df,
            left_on="recommended_movie",
            right_on="title",
            how="left",
        )
        # Create new rating entries from feedback
        feedback_entries = feedback_merged[["movieId", "user_rating"]].copy()
        feedback_entries["userId"] = virtual_user_id
        feedback_entries["timestamp"] = pd.Timestamp.now()
        # Rename the user_rating column to rating for consistency
        feedback_entries = feedback_entries.rename(columns={"user_rating": "rating"})
        # Append feedback entries to the original merged data
        merged = pd.concat([merged, feedback_entries], ignore_index=True)
    else:
        print("No feedback found; using original data only.")

    # Create an updated pivot table
    # Filter to movies with at least 50 ratings (lower threshold to account for new feedback)
    ratings_count = merged.groupby("title")["rating"].count()
    popular_movies = ratings_count[ratings_count >= 50].index
    filtered_data = merged[merged["title"].isin(popular_movies)]
    pivot = filtered_data.pivot_table(index="userId", columns="title", values="rating")
    pivot_filled = pivot.fillna(0)

    # Train NMF model on the updated pivot table
    nmf_model = NMF(n_components=n_components, init="random", random_state=42)
    W = nmf_model.fit_transform(pivot_filled)
    H = nmf_model.components_

    # Save the dynamic model data for later use
    model_data = {"nmf_model": nmf_model, "pivot": pivot, "W": W, "H": H}
    joblib.dump(model_data, output_model)
    print(f"Dynamic model updated and saved to {output_model}")
    return model_data


if __name__ == "__main__":
    update_dynamic_model()

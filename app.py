import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
from advanced_recommender import create_pivot_table, advanced_recommendations
from data_preprocessing import load_movies
from logger import logger  # Import custom logger


@st.cache_resource
def get_advanced_pivot():
    # Create the pivot table for recommendations using the advanced recommender's function.
    logger.info("Generating advanced pivot table...")
    pivot = create_pivot_table()
    logger.info("Pivot table generated successfully.")
    return pivot


def save_feedback(feedback_data, file_path="feedback.csv"):
    """
    Saves user feedback to a CSV file.
    If the file exists, appends the new feedback; otherwise, creates a new file.
    """
    logger.info("Saving feedback data...")
    df_feedback = pd.DataFrame(feedback_data)
    if os.path.exists(file_path):
        df_feedback.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df_feedback.to_csv(file_path, mode="w", header=True, index=False)
    logger.info("Feedback saved to %s", file_path)


def main():
    st.title("Movie Recommendation Dashboard (Advanced with Feedback)")
    st.write(
        "Select a movie from the dropdown below to see similar recommendations using matrix factorization. After viewing recommendations, please provide feedback on each one."
    )
    logger.info("Dashboard launched.")

    # Load the pivot table from the advanced recommender
    pivot = get_advanced_pivot()

    # Get list of movies from the pivot table columns and sort alphabetically
    movie_list = list(pivot.columns)
    movie_list.sort()
    selected_movie = st.selectbox("Choose a movie", movie_list)
    logger.info("User selected movie: %s", selected_movie)

    if st.button("Get Advanced Recommendations"):
        try:
            logger.info("Generating advanced recommendations for: %s", selected_movie)
            recommendations = advanced_recommendations(selected_movie, pivot)
            st.write(f"Movies similar to **{selected_movie}**:")
            rec_df = recommendations.reset_index().rename(
                columns={selected_movie: "Similarity Score"}
            )
            st.table(rec_df)
            logger.info("Recommendations generated successfully.")

            # Display feedback form using st.form to avoid re-running on slider change
            st.markdown("### Provide Feedback on the Recommendations")
            with st.form(key="feedback_form"):
                feedback = []
                # For each recommendation, create a slider inside the form.
                for idx, row in rec_df.iterrows():
                    rec_movie = row["index"]
                    rating = st.slider(
                        f"Rate '{rec_movie}' (1=poor, 5=excellent):",
                        min_value=1,
                        max_value=5,
                        key=f"slider_{idx}",
                    )
                    feedback.append(
                        {
                            "selected_movie": selected_movie,
                            "recommended_movie": rec_movie,
                            "similarity_score": row["Similarity Score"],
                            "user_rating": rating,
                            "timestamp": datetime.datetime.now().isoformat(),
                        }
                    )
                submitted = st.form_submit_button("Submit Feedback")
                if submitted:
                    save_feedback(feedback)
                    st.success("Feedback submitted successfully!")
                    logger.info("Feedback submitted for movie: %s", selected_movie)
        except Exception as e:
            st.error(f"Error: {e}")
            logger.error("Error generating recommendations: %s", e)

    st.markdown("---")
    st.write("When you're done, you can exit the dashboard using the button below.")
    if st.button("Exit Dashboard"):
        st.write("Exiting dashboard...")
        logger.info("Dashboard exit triggered by user.")
        os._exit(0)


if __name__ == "__main__":
    main()

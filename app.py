import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from advanced_recommender import create_pivot_table, advanced_recommendations
from recommendation_engine import (
    create_pivot_table as create_pivot_table_traditional,
    get_recommendations,
    compute_similarity,
)
from data_preprocessing import load_movies
from logger import logger


@st.cache_resource
def get_pivot_advanced():
    # Create the pivot table for advanced recommendations.
    logger.info("Generating advanced pivot table...")
    pivot = create_pivot_table()
    logger.info("Pivot table generated successfully.")
    return pivot


@st.cache_resource
def get_pivot_traditional():
    # Create the pivot table for traditional recommendations.
    pivot = create_pivot_table_traditional()
    return pivot


def plot_feedback_trends(feedback_file="feedback.csv"):
    if os.path.exists(feedback_file):
        feedback = pd.read_csv(feedback_file)
        st.subheader("Average Rating per Recommended Movie")
        # Plot average rating per recommended movie
        avg_ratings = (
            feedback.groupby("recommended_movie")["user_rating"].mean().reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x="recommended_movie",
            y="user_rating",
            data=avg_ratings,
            ax=ax,
            color="skyblue",
        )
        ax.set_ylabel("Average Rating")
        ax.set_xlabel("Recommended Movie")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)

        st.subheader("Feedback Count Over Time")
        # Plot feedback count over time
        feedback["timestamp"] = pd.to_datetime(feedback["timestamp"])
        feedback["date"] = feedback["timestamp"].dt.date
        daily_counts = feedback.groupby("date").size().reset_index(name="count")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.lineplot(x="date", y="count", data=daily_counts, marker="o", ax=ax2)
        ax2.set_ylabel("Feedback Count")
        ax2.set_xlabel("Date")
        st.pyplot(fig2)
    else:
        st.info("No feedback data available yet.")


def main():
    st.title("Enhanced Movie Recommendation Dashboard")
    st.write(
        "Explore movie recommendations and analytics with this enhanced dashboard."
    )

    # Display current working directory for debugging purposes
    cwd = os.getcwd()
    st.write("Current working directory:", cwd)
    logger.info("Current working directory: %s", cwd)

    tabs = st.tabs(["Recommendations", "Feedback Trends", "A/B Testing"])

    # Tab 1: Recommendations (Advanced with Feedback)
    with tabs[0]:
        st.header("Advanced Recommendations")
        pivot = get_pivot_advanced()
        movie_list = list(pivot.columns)
        movie_list.sort()
        selected_movie = st.selectbox("Choose a movie", movie_list)

        if st.button("Get Advanced Recommendations", key="advanced"):
            try:
                logger.info(
                    "Generating advanced recommendations for: %s", selected_movie
                )
                recommendations = advanced_recommendations(selected_movie, pivot)
                st.write(f"Movies similar to **{selected_movie}**:")
                rec_df = recommendations.reset_index().rename(
                    columns={selected_movie: "Similarity Score"}
                )
                st.table(rec_df)
                logger.info("Recommendations generated successfully.")

                # Display feedback form using st.form to avoid re-runs on slider change
                st.markdown("### Provide Feedback on the Recommendations")
                with st.form(key="feedback_form"):
                    feedback = []
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
                        # Save feedback using an absolute path
                        feedback_file = os.path.join(os.getcwd(), "feedback.csv")
                        logger.info("Saving feedback to: %s", feedback_file)
                        st.write("Saving feedback to:", feedback_file)
                        try:
                            df_feedback = pd.DataFrame(feedback)
                            header_needed = not os.path.exists(feedback_file)
                            df_feedback.to_csv(
                                feedback_file,
                                mode="a",
                                header=header_needed,
                                index=False,
                            )
                            st.success(
                                f"Feedback submitted successfully! Saved to {feedback_file}"
                            )
                            logger.info(
                                "Feedback submitted for movie: %s", selected_movie
                            )
                        except Exception as ex:
                            st.error(f"Error saving feedback: {ex}")
                            logger.error("Error saving feedback: %s", ex)
            except Exception as e:
                st.error(f"Error: {e}")
                logger.error("Error generating recommendations: %s", e)

    # Tab 2: Feedback Trends
    with tabs[1]:
        st.header("User Feedback Trends")
        plot_feedback_trends()

    # Tab 3: A/B Testing (Traditional vs Advanced)
    with tabs[2]:
        st.header("A/B Testing: Traditional vs Advanced Recommendations")
        pivot_trad = get_pivot_traditional()
        movie_list_trad = list(pivot_trad.columns)
        movie_list_trad.sort()
        selected_movie_ab = st.selectbox(
            "Choose a movie for A/B testing", movie_list_trad, key="ab"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Traditional Recommendations")
            try:
                corr_matrix = compute_similarity(pivot_trad)
                trad_recs = get_recommendations(
                    selected_movie_ab, pivot_trad, corr_matrix
                )
                trad_df = trad_recs.reset_index().rename(
                    columns={selected_movie_ab: "Correlation"}
                )
                st.table(trad_df)
            except Exception as e:
                st.error(f"Error (Traditional): {e}")
        with col2:
            st.subheader("Advanced Recommendations")
            try:
                adv_recs = advanced_recommendations(
                    selected_movie_ab, pivot_trad
                )  # using same pivot for comparison
                adv_df = adv_recs.reset_index().rename(
                    columns={selected_movie_ab: "Similarity Score"}
                )
                st.table(adv_df)
            except Exception as e:
                st.error(f"Error (Advanced): {e}")

    st.markdown("---")
    st.write("When you're done, click the button below to exit the dashboard.")
    if st.button("Exit Dashboard"):
        st.write("Exiting dashboard...")
        logger.info("Dashboard exit triggered by user.")
        os._exit(0)


if __name__ == "__main__":
    main()

import sys
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Import project modules
from advanced_recommender import create_pivot_table, advanced_recommendations
from recommendation_engine import (
    create_pivot_table as create_pivot_table_traditional,
    get_recommendations,
    compute_similarity,
)
from data_preprocessing import merge_data, load_movies
from logger import logger


# --------------------------------------------------
# Global pivot functions for TEXT MODE (uncached)
# --------------------------------------------------
def get_pivot_advanced_global():
    logger.info("Generating advanced pivot table (global)...")
    pivot = create_pivot_table()
    logger.info("Advanced pivot table generated successfully (global).")
    return pivot


def get_pivot_traditional_global():
    logger.info("Generating traditional pivot table (global)...")
    pivot = create_pivot_table_traditional()
    logger.info("Traditional pivot table generated successfully (global).")
    return pivot


# ------------------------------
# STREAMLIT DASHBOARD FUNCTIONS
# ------------------------------
def run_streamlit_app():
    import streamlit as st

    st.title("Enhanced Movie Recommendation Dashboard")
    st.write(
        "Explore movie recommendations, feedback analytics, and A/B testing of recommendation strategies."
    )

    # Display current working directory for verification
    cwd = os.getcwd()
    st.write("Current working directory:", cwd)
    logger.info("Current working directory: %s", cwd)

    @st.cache_resource
    def get_pivot_advanced_streamlit():
        return get_pivot_advanced_global()

    @st.cache_resource
    def get_pivot_traditional_streamlit():
        return get_pivot_traditional_global()

    tabs = st.tabs(["Recommendations", "Feedback Trends", "A/B Testing"])

    # Tab 1: Advanced Recommendations with Feedback
    with tabs[0]:
        st.header("Advanced Recommendations")
        pivot = get_pivot_advanced_streamlit()
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
                        feedback_file = os.path.join(os.getcwd(), "feedback.csv")
                        logger.info("Saving feedback to: %s", feedback_file)
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
    def plot_feedback_trends(feedback_file="feedback.csv"):
        if os.path.exists(feedback_file):
            feedback = pd.read_csv(feedback_file)
            st.subheader("Average Rating per Recommended Movie")
            avg_ratings = (
                feedback.groupby("recommended_movie")["user_rating"]
                .mean()
                .reset_index()
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

    with tabs[1]:
        st.header("User Feedback Trends")
        plot_feedback_trends()

    # Tab 3: A/B Testing (Traditional vs Advanced)
    with tabs[2]:
        st.header("A/B Testing: Traditional vs Advanced Recommendations")
        pivot_trad = get_pivot_traditional_streamlit()
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
                adv_recs = advanced_recommendations(selected_movie_ab, pivot_trad)
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


# ------------------------------
# TEXT-BASED MENU FUNCTIONS
# ------------------------------
def run_text_menu():
    console = Console()
    while True:
        console.print(
            Panel.fit(
                Text("Movie Recommendation System", style="bold magenta"),
                title="Main Menu",
                border_style="magenta",
            )
        )
        console.print("Select an option:")
        console.print("1. Get movie recommendations (traditional)")
        console.print("2. View sample merged data")
        console.print("3. Launch interactive dashboard (Streamlit)")
        console.print("4. Get advanced recommendations (Matrix Factorization)")
        console.print("5. Update dynamic model (incorporate feedback)")
        console.print("6. Run with Streamlit directly")
        console.print("7. Exit")
        choice = input("Enter your choice (1/2/3/4/5/6/7): ").strip()
        if choice == "1":
            terminal_recommendation(console)
        elif choice == "2":
            console.print(
                Panel.fit(
                    Text("Loading merged data...", style="bold blue"),
                    title="Merged Data",
                    border_style="blue",
                )
            )
            data = merge_data()
            console.print(data.head())
        elif choice == "3":
            console.print(
                Panel.fit(
                    Text("Launching Streamlit Dashboard...", style="bold green"),
                    title="Dashboard",
                    border_style="green",
                )
            )
            os.system("STREAMLIT_MODE=1 streamlit run main.py")
        elif choice == "4":
            terminal_advanced_recommendation(console)
        elif choice == "5":
            try:
                from dynamic_update import update_dynamic_model

                console.print(
                    Panel.fit(
                        Text(
                            "Updating dynamic model using feedback...",
                            style="bold blue",
                        ),
                        title="Dynamic Update",
                        border_style="blue",
                    )
                )
                update_dynamic_model()
                console.print(
                    Panel.fit(
                        Text("Dynamic model updated successfully.", style="bold green"),
                        title="Success",
                        border_style="green",
                    )
                )
            except Exception as e:
                console.print(
                    Panel.fit(
                        Text(f"Error updating dynamic model: {e}", style="bold red"),
                        title="Error",
                        border_style="red",
                    )
                )
        elif choice == "6":
            console.print(
                Panel.fit(
                    Text("Launching Streamlit directly...", style="bold green"),
                    title="Streamlit Mode",
                    border_style="green",
                )
            )
            os.system("STREAMLIT_MODE=1 streamlit run main.py")
        elif choice == "7":
            console.print(
                Panel.fit(
                    Text("Goodbye!", style="bold blue"),
                    title="Exit",
                    border_style="blue",
                )
            )
            sys.exit(0)
        else:
            console.print(
                Panel.fit(
                    Text("Invalid choice. Please try again.", style="bold red"),
                    title="Error",
                    border_style="red",
                )
            )


def terminal_recommendation(console):
    console.print(
        Panel.fit(
            Text("Generating traditional recommendations...", style="bold blue"),
            title="Engine",
            border_style="blue",
        )
    )
    pivot = create_pivot_table_traditional()
    corr_matrix = compute_similarity(pivot)
    movie_list = list(pivot.columns)
    console.print(
        "\nEnter a movie title for recommendations (partial titles accepted):"
    )
    movie_query = input("Movie Title: ").strip()
    from thefuzz import process

    match, score = process.extractOne(movie_query, movie_list)
    best_match = match if score >= 70 else None
    if best_match is None:
        console.print(
            Panel.fit(
                "No close match found for your query. Please try again.",
                style="bold red",
                border_style="red",
            )
        )
        return
    console.print(
        Panel.fit(
            f"Best match found: [bold green]{best_match}[/bold green]",
            border_style="green",
        )
    )
    try:
        recommendations = get_recommendations(best_match, pivot, corr_matrix)
        console.print(
            Panel.fit(
                f"Recommendations for '{best_match}':",
                style="bold green",
                border_style="green",
            )
        )
        for title, corr in recommendations.items():
            console.print(f"{title}: Correlation = {corr:.2f}")
    except Exception as e:
        console.print(Panel.fit(f"Error: {e}", style="bold red", border_style="red"))


def terminal_advanced_recommendation(console):
    console.print(
        Panel.fit(
            Text("Generating advanced recommendations using NMF...", style="bold blue"),
            title="Advanced Engine",
            border_style="blue",
        )
    )
    pivot = get_pivot_advanced_global()  # use global function for text mode
    movie_list = list(pivot.columns)
    console.print(
        "\nEnter a movie title for advanced recommendations (partial titles accepted):"
    )
    movie_query = input("Movie Title: ").strip()
    from thefuzz import process

    match, score = process.extractOne(movie_query, movie_list)
    best_match = match if score >= 70 else None
    if best_match is None:
        console.print(
            Panel.fit(
                "No close match found for your query. Please try again.",
                style="bold red",
                border_style="red",
            )
        )
        return
    console.print(
        Panel.fit(
            f"Best match found: [bold green]{best_match}[/bold green]",
            border_style="green",
        )
    )
    try:
        recommendations = advanced_recommendations(best_match, pivot)
        console.print(
            Panel.fit(
                f"Advanced Recommendations for '{best_match}':",
                style="bold green",
                border_style="green",
            )
        )
        for title, score in recommendations.items():
            console.print(f"{title}: Similarity Score = {score:.2f}")
    except Exception as e:
        console.print(Panel.fit(f"Error: {e}", style="bold red", border_style="red"))


# ------------------------------
# MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":
    # Check if the environment variable STREAMLIT_MODE is set.
    if os.environ.get("STREAMLIT_MODE") == "1":
        run_streamlit_app()
    else:
        run_text_menu()

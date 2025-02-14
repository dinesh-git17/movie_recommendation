# main.py
import os
import sys
from data_preprocessing import merge_data
from recommendation_engine import (
    create_pivot_table,
    compute_similarity,
    get_recommendations,
)
from advanced_recommender import (
    advanced_recommendations,
    create_pivot_table as create_pivot_table_advanced,
)
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from thefuzz import process  # For fuzzy matching


def find_movie_match(query, movie_list, threshold=70):
    """
    Uses fuzzy matching to find the best movie match.
    Returns the best matching movie title if the score is above threshold.
    """
    match, score = process.extractOne(query, movie_list)
    return match if score >= threshold else None


def terminal_recommendation(console):
    console.print(
        Panel.fit(
            Text("Generating traditional recommendations...", style="bold blue"),
            title="Engine",
            border_style="blue",
        )
    )
    pivot = create_pivot_table()
    corr_matrix = compute_similarity(pivot)
    movie_list = list(pivot.columns)
    console.print("\nEnter a movie title (partial titles accepted):")
    movie_query = input("Movie Title: ").strip()
    best_match = find_movie_match(movie_query, movie_list)
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
    pivot = create_pivot_table_advanced()  # Same as original pivot function
    movie_list = list(pivot.columns)
    console.print("\nEnter a movie title (partial titles accepted):")
    movie_query = input("Movie Title: ").strip()
    best_match = find_movie_match(movie_query, movie_list)
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


def view_merged_data(console):
    console.print(
        Panel.fit(
            Text("Loading merged data...", style="bold blue"),
            title="Merged Data",
            border_style="blue",
        )
    )
    data = merge_data()
    console.print(data.head())


def launch_dashboard(console):
    console.print(
        Panel.fit(
            Text("Launching Streamlit Dashboard...", style="bold green"),
            title="Dashboard",
            border_style="green",
        )
    )
    os.system("streamlit run app.py")


def update_dynamic(console):
    console.print(
        Panel.fit(
            Text("Updating dynamic model using feedback...", style="bold blue"),
            title="Dynamic Update",
            border_style="blue",
        )
    )
    try:
        from dynamic_update import update_dynamic_model

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


def main_menu():
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
        console.print("6. Exit")
        choice = input("Enter your choice (1/2/3/4/5/6): ").strip()
        if choice == "1":
            terminal_recommendation(console)
        elif choice == "2":
            view_merged_data(console)
        elif choice == "3":
            launch_dashboard(console)
        elif choice == "4":
            terminal_advanced_recommendation(console)
        elif choice == "5":
            update_dynamic(console)
        elif choice == "6":
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


if __name__ == "__main__":
    main_menu()

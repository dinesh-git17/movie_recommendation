Movie Recommendation System
============================

Overview:
---------
This project is an advanced movie recommendation system built using the MovieLens 100K dataset.
It demonstrates various machine learning techniques including:
  - Data preprocessing and merging of the original MovieLens files (e.g., u.data, u.item).
  - Traditional recommendation using Pearson correlation on a user-rating pivot table.
  - Advanced recommendation using matrix factorization (Non-negative Matrix Factorization, NMF)
    to extract latent factors and compute cosine similarities.
  - Fuzzy matching to allow users to input partial or approximate movie titles.
  - A user feedback loop that captures ratings on recommended movies and dynamically updates
    the model over time.
  - An interactive Streamlit dashboard for real-time recommendations and feedback.
  - Enhanced logging with colorful, emoji-enhanced messages.

Project Features:
-----------------
1. Data Preprocessing:
   - Loads and merges MovieLens data from the original file formats.
   - Prepares a unified dataset with userId, movieId, rating, timestamp, and movie title.

2. Recommendation Engines:
   - Traditional Recommendation: Uses a pivot table and Pearson correlation to find similar movies.
   - Advanced Recommendation: Uses NMF-based matrix factorization and cosine similarity for improved recommendations.
   - Fuzzy Matching: Accepts partial movie titles and finds the closest match using fuzzy logic.

3. Dynamic Model Updates:
   - Captures user feedback from the dashboard (ratings on recommendations).
   - Incorporates feedback as additional ratings and retrains the NMF model to adapt over time.

4. Interactive Dashboard:
   - Built with Streamlit, it allows users to select a movie, view recommendations, and submit feedback via a form.
   - Uses st.form to prevent unnecessary re-runs during slider adjustments.
   - Includes an “Exit Dashboard” button to return control to the terminal.

5. Logging:
   - A custom logger provides colorful, emoji-enhanced logging messages to track key events
     (e.g., dashboard launch, recommendation generation, feedback submission).

Project Structure:
------------------
- data_preprocessing.py   : Loads and preprocesses the MovieLens dataset.
- recommendation_engine.py: Implements traditional recommendation logic.
- advanced_recommender.py : Implements advanced recommendations using NMF.
- dynamic_update.py       : Incorporates user feedback and updates the model dynamically.
- app.py                  : Streamlit dashboard for interactive recommendations and feedback.
- main.py                 : Unified main file offering a text-based menu for all components.
- logger.py               : Custom logger module with colorful, emoji-enhanced logging.
- README.txt              : This documentation file.
- feedback.csv            : (Generated at runtime) Stores user feedback.

Setup Instructions:
-------------------
1. Download the MovieLens 100K dataset from:
   https://grouplens.org/datasets/movielens/100k/
   Extract the files (e.g., u.data, u.item) into a folder named "data" in the project root.

2. Install the required Python packages by running:
       pip install -r requirements.txt
   (Ensure your requirements.txt includes: pandas, numpy, scikit-learn, streamlit, joblib, colorlog, thefuzz[speedup].)

3. Run the Unified Main File:
   From the terminal, run:
       python main.py
   This will present a menu with options for traditional recommendations, advanced recommendations,
   viewing sample data, launching the dashboard, and updating the model with feedback.

4. Run the Streamlit Dashboard:
   Alternatively, launch the interactive dashboard with:
       streamlit run app.py
   Use the dashboard to select a movie, view recommendations, and provide feedback.

Usage:
------
- In the terminal (via main.py), you can:
  1. Get traditional recommendations based on correlation.
  2. View a sample of the merged MovieLens data.
  3. Launch the interactive Streamlit dashboard.
  4. Get advanced recommendations using NMF-based matrix factorization.
  5. Update the dynamic model to incorporate new user feedback.
  6. Exit the program.

- The Streamlit dashboard uses fuzzy matching so users can type partial movie titles.
  After recommendations are shown, users can rate each suggestion via sliders within a form,
  and their feedback is stored in feedback.csv for dynamic model updates.

Future Enhancements:
--------------------
- Develop a hybrid recommender that combines collaborative and content-based filtering.
- Deploy the system as a REST API using Flask or FastAPI.
- Enhance the dashboard with additional visualizations and user analytics.
- Integrate online learning algorithms to update the model continuously.
- Implement CI/CD pipelines and containerize the project for production deployment.

Contact:
--------
For questions, suggestions, or contributions, please open an issue or submit a pull request on GitHub.


Thank you for exploring the Movie Recommendation System project!

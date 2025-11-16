🌍 Hybrid Travel Planner Recommendation System

Live Demo: https://hybridtravelplanner.streamlit.app/

📌 Overview

Travel planning is complex — users have unique interests, family needs, and varied travel histories.
This project builds an AI-powered Hybrid Travel Recommendation System that combines:

✔ Machine Learning (Regression + Classification)
✔ Natural Language Processing (TF-IDF + Content Similarity)
✔ User demographics & preferences
✔ Destination metadata

The final system recommends personalized travel destinations for each user based on:

predicted satisfaction rating

probability of visiting the destination

preference alignment using NLP

destination popularity and attributes

All results are integrated into a single HybridScore and displayed through a clean Streamlit web app.

🚀 Key Features
🔹 1. Multi-Model Hybrid Recommendation

The system integrates three different ML modules:

A. Rating Prediction — Ridge Regression

Predicts how much a user will like a destination based on demographics, destination attributes, and TF-IDF features.

B. Visit Probability — Random Forest Classifier

Estimates how likely a user is to visit a destination using popularity, user type, and similarity features.

C. NLP-Based Content Similarity — TF-IDF + Cosine Similarity

Matches user textual preferences (Beaches, Nature, Adventure, Historical…) with destination descriptions.

These three signals combine to produce the final HybridScore.

🔹 2. End-to-End ML Pipeline

The project includes:

dataset preprocessing

feature engineering

one-hot encoding

TF-IDF vectorization

handling missing values

model training, evaluation, saving (joblib)

hybrid scoring

This mirrors the workflow used in real-world recommender systems.

🔹 3. Modern Streamlit UI

The deployed app provides:

user selection

adjustable number of recommendations

HybridScore ranking

detailed destination insights

fully responsive design

Live App: https://hybridtravelplanner.streamlit.app/

📂 Project Structure
📁 HybridTravelPlanner
│
├── app.py                               # Streamlit web app
├── ridge_experience_model.joblib        # Saved regression model
├── visit_probability_model.joblib       # Saved classification model
│
├── Final_Updated_Expanded_Users.csv     # User data
├── Final_Updated_Expanded_UserHistory.csv
├── Final_Updated_Expanded_Reviews.csv
├── Expanded_Destinations.csv            # Destination dataset
│
└── README.md                            # Project documentation

📊 Algorithms & Techniques Used
🧠 Machine Learning

Ridge Regression

Random Forest Classifier

Hyperparameter tuning

Train-test split

Feature importance analysis

Regularization (L2)

Bias-variance considerations

📝 Natural Language Processing

TF-IDF Vectorizer

Cosine similarity

Text normalization

📦 Data Engineering

One-hot encoding

Merging multi-table datasets

Handling numerical + categorical features

Vector concatenation (34-feature regression input)

💡 Hybrid Recommendation Strategy

Final score =

HybridScore = 0.4 * PredRating  
              + 0.3 * VisitProbability  
              + 0.3 * ContentSimilarity


Values are normalized for fairness.

📄 Dataset Summary
Users Dataset

User demographics

Gender

Travel preferences

Number of children

Destinations Dataset

DestinationID

Name

Type (Beach, Nature, Historical, etc.)

Best time to visit

Popularity score

Travel History

Past destinations visited

Ratings

Useful for modeling user behavior

Reviews Dataset

Additional rating labels

Helps overcome sparse-rating problem

📊 Visual Insights Included

The notebook includes multiple visualizations:

Distribution of destination popularity

User preference breakdown

TF-IDF similarity heatmaps

Random Forest feature importance

Predicted rating distribution

These graphs improve interpretability and project presentation.

▶️ How to Run Locally
1. Install dependencies
pip install -r requirements.txt

2. Run the app
streamlit run app.py


The app will launch on http://localhost:8501/.

🌐 Deployment

The project is deployed using Streamlit Cloud, enabling public access.

Live Demo: https://hybridtravelplanner.streamlit.app/

🏁 Conclusion

This project demonstrates how Machine Learning, NLP, and User Modelling can come together to build a practical, real-world recommendation system.
The hybrid approach ensures:

better personalization

higher accuracy

transparent scoring

real-time recommendation generation

It serves as a strong portfolio project for roles in Data Science, Machine Learning, AI Engineering, and Data Analytics.

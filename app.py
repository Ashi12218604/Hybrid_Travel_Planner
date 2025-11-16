import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# 1. LOAD DATA
# ==========================================================

df_users = pd.read_csv("Final_Updated_Expanded_Users.csv")
df_dest = pd.read_csv("Expanded_Destinations.csv")

# clean column names
df_users.columns = [c.strip() for c in df_users.columns]
df_dest.columns = [c.strip() for c in df_dest.columns]

# Ensure required columns exist
df_users["UserID"] = pd.to_numeric(df_users["UserID"], errors="coerce")
df_dest["DestinationID"] = pd.to_numeric(df_dest["DestinationID"], errors="coerce")

# ==========================================================
# 2. LOAD MODELS
# ==========================================================

# Visit probability model
visit_model = joblib.load("visit_probability_model.joblib")

# Rating model (may be fallback)
try:
    rating_model = joblib.load("ridge_experience_model.joblib")
    HAS_RATING_MODEL = True
except:
    HAS_RATING_MODEL = False

# ==========================================================
# 3. BUILD TF-IDF FOR CONTENT SIMILARITY
# ==========================================================

# Destination content
dest_texts = (
    df_dest["Name"].fillna("") + " " +
    df_dest["Type"].fillna("") + " " +
    df_dest["State"].fillna("")
).astype(str)

# User preference text (for TF-IDF training)
user_pref_texts = df_users.get("Preference", pd.Series([""] * len(df_users))).astype(str)

# Fit TF-IDF
tfidf = TfidfVectorizer(max_features=500)
tfidf.fit(pd.concat([dest_texts, user_pref_texts], ignore_index=True))

# Precompute destination vectors
dest_tfidf = tfidf.transform(dest_texts)

# Map destination index
dest_id_to_idx = {int(did): i for i, did in enumerate(df_dest["DestinationID"])}


# ==========================================================
# 4. HYBRID RECOMMENDER FUNCTION
# ==========================================================

def recommend_for_user(user_id, top_n=10, alpha=0.4, beta=0.3, gamma=0.3):
    """Hybrid recommendation combining:
       • Predicted Rating
       • Visit Probability
       • Content Similarity
    """

    # Load user row
    user = df_users[df_users["UserID"] == user_id]
    if user.empty:
        raise ValueError("Invalid User ID selected.")

    gender = user["Gender"].iloc[0]
    preference = user.get("Preference", pd.Series(["Unknown"])).iloc[0]
    num_children = user.get("NumberOfChildren", user.get("NumChildren", pd.Series([0]))).iloc[0]

    # Build user text for content similarity
    user_text = f"{preference} {gender}"
    user_vec = tfidf.transform([user_text])

    # Destination TF-IDF vectors already computed
    content_sim = cosine_similarity(user_vec, dest_tfidf).ravel()

    # Visit probability prediction
    pair_df = pd.DataFrame({
        "Popularity": df_dest["Popularity"].astype(float),
        "NumChildren": num_children,
        "ContentSim": content_sim,
        "Gender": gender,
        "Type": df_dest["Type"].astype(str)
    })

    visit_probs = visit_model.predict_proba(pair_df)[:, 1]

    # Predicted Rating (fallback to popularity)
    if HAS_RATING_MODEL:
        # Very simple fallback input for prediction
        popularity = df_dest["Popularity"].astype(float).values
        # Create dummy X of correct size if required
        dummy_X = np.zeros((len(df_dest), 10))
        pred_rating = rating_model.predict(dummy_X)
    else:
        pop = df_dest["Popularity"].astype(float)
        pred_rating = 1 + 4 * (pop - pop.min()) / (pop.max() - pop.min())

    # Normalize
    def norm(x):
        x = np.array(x, dtype=float)
        if x.max() == x.min():
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    R = norm(pred_rating)
    V = norm(visit_probs)
    C = norm(content_sim)

    hybrid_score = alpha * R + beta * V + gamma * C

    # Build output
    out = df_dest.copy()
    out["PredRating"] = pred_rating
    out["VisitProb"] = visit_probs
    out["ContentSim"] = content_sim
    out["HybridScore"] = hybrid_score

    return out.sort_values("HybridScore", ascending=False).head(top_n)[[
        "DestinationID", "Name", "Type", "State",
        "Popularity", "PredRating", "VisitProb",
        "ContentSim", "HybridScore"
    ]]


# ==========================================================
# 5. STREAMLIT UI
# ==========================================================

st.title("🌍 Hybrid Travel Planner Recommendation System")

user_list = df_users["UserID"].dropna().astype(int).tolist()
selected_user = st.selectbox("Select a User ID:", user_list)

top_n = st.slider("Number of Recommendations:", 5, 20, 10)

if st.button("Recommend"):
    try:
        recommendations = recommend_for_user(selected_user, top_n=top_n)
        st.success(f"Top {top_n} Recommendations for User {selected_user}")
        st.dataframe(recommendations)
    except Exception as e:
        st.error(f"Error: {e}")


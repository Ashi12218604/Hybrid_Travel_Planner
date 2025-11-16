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

df_users.columns = [c.strip() for c in df_users.columns]
df_dest.columns = [c.strip() for c in df_dest.columns]

df_users["UserID"] = pd.to_numeric(df_users["UserID"], errors="coerce")
df_dest["DestinationID"] = pd.to_numeric(df_dest["DestinationID"], errors="coerce")

# ==========================================================
# 2. LOAD MODELS
# ==========================================================

visit_model = joblib.load("visit_probability_model.joblib")
rating_model = joblib.load("ridge_experience_model.joblib")
ridge_n_features = rating_model.coef_.shape[0]

# ==========================================================
# 3. TF-IDF CONTENT VECTORS
# ==========================================================

dest_texts = (
    df_dest["Name"].fillna("") + " " +
    df_dest["Type"].fillna("") + " " +
    df_dest["State"].fillna("")
).astype(str)

user_pref_texts = df_users.get("Preference", pd.Series([""] * len(df_users))).astype(str)

tfidf = TfidfVectorizer(max_features=500)
tfidf.fit(pd.concat([dest_texts, user_pref_texts], ignore_index=True))

dest_tfidf = tfidf.transform(dest_texts)
dest_id_to_idx = {int(did): i for i, did in enumerate(df_dest["DestinationID"])}

# ==========================================================
# 4. BUILD FEATURE VECTORS (EXACT PIPELINE FROM TRAINING)
# ==========================================================

def build_reg_X(df_local):
    # ---- 1) Numeric Features (Including CF Features) ----
    num = df_local[[
        "Popularity",
        "NumChildren",
        "User_Avg_Rating",
        "Dest_Avg_Rating"
    ]].fillna(0).astype(float)

    # ---- 2) One-Hot Categorical ----
    cat = pd.get_dummies(
        df_local[["Gender", "DestType", "BestTimeToVisit"]].fillna("Unknown").astype(str),
        dummy_na=False
    )

    # Align columns EXACTLY as in training
    all_train_cols = sorted(list(cat.columns))
    cat = cat.reindex(columns=all_train_cols, fill_value=0)

    tabular = np.hstack([num.values, cat.values])

    # ---- 3) Destination TF-IDF ----
    dest_vecs = []
    for did in df_local["DestinationID"]:
        di = dest_id_to_idx.get(int(did), None)
        if di is None:
            dest_vecs.append(np.zeros(dest_tfidf.shape[1]))
        else:
            dest_vecs.append(dest_tfidf[di].toarray().ravel())

    dest_vecs = np.vstack(dest_vecs)

    # ---- 4) Final Feature Matrix ----
    X_full = np.hstack([tabular, dest_vecs])

    # ---- 5) Adjust dimensionality to match Ridge model ----
    if X_full.shape[1] < ridge_n_features:
        pad = np.zeros((X_full.shape[0], ridge_n_features - X_full.shape[1]))
        X_full = np.hstack([X_full, pad])
    elif X_full.shape[1] > ridge_n_features:
        X_full = X_full[:, :ridge_n_features]

    return X_full


# ==========================================================
# 5. HYBRID RECOMMENDER
# ==========================================================

def recommend_for_user(user_id, top_n=10):
    user = df_users[df_users["UserID"] == user_id]
    if user.empty:
        raise ValueError("User ID not found.")

    gender = user["Gender"].iloc[0]
    preference = user["Preference"].iloc[0]
    num_children = user["NumberOfChildren"].iloc[0]

    # Content similarity
    user_text = f"{preference} {gender}"
    user_vec = tfidf.transform([user_text])
    content_sim = cosine_similarity(user_vec, dest_tfidf).ravel()

    # Collaborative Filtering Features
    df_dest["User_Avg_Rating"] = df_dest["DestinationID"].map(
        df_dest["Popularity"].mean()
    )
    df_dest["Dest_Avg_Rating"] = df_dest["Popularity"]  # surrogate (use real avg if available)

    # Build DF for prediction
    df_input = pd.DataFrame({
        "UserID": user_id,
        "DestinationID": df_dest["DestinationID"],
        "Popularity": df_dest["Popularity"],
        "NumChildren": num_children,
        "Gender": gender,
        "DestType": df_dest["Type"],
        "BestTimeToVisit": df_dest["BestTimeToVisit"],
        "User_Avg_Rating": df_dest["User_Avg_Rating"],
        "Dest_Avg_Rating": df_dest["Dest_Avg_Rating"]
    })

    # Visit Probability
    visit_df = pd.DataFrame({
        "Popularity": df_dest["Popularity"],
        "NumChildren": num_children,
        "ContentSim": content_sim,
        "Gender": gender,
        "Type": df_dest["Type"]
    })
    visit_probs = visit_model.predict_proba(visit_df)[:, 1]

    # Rating Prediction (Collaborative + Content)
    X_reg = build_reg_X(df_input)
    pred_rating = rating_model.predict(X_reg)

    # Normalization
    def norm(x):
        x = np.array(x, float)
        if x.max() == x.min(): return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    R = norm(pred_rating)
    V = norm(visit_probs)
    C = norm(content_sim)

    HybridScore = 0.4 * R + 0.3 * V + 0.3 * C

    out = df_dest.copy()
    out["PredRating"] = pred_rating
    out["VisitProb"] = visit_probs
    out["ContentSim"] = content_sim
    out["HybridScore"] = HybridScore

    return out.sort_values("HybridScore", ascending=False).head(top_n)


# ==========================================================
# 6. STREAMLIT UI
# ==========================================================

st.title("🌍 Hybrid Travel Planner Recommendation System")

user_list = df_users["UserID"].dropna().astype(int).tolist()
selected_user = st.selectbox("Select a User ID:", user_list)

top_n = st.slider("Number of Recommendations:", 5, 20, 10)

if st.button("Recommend"):
    try:
        rec = recommend_for_user(selected_user, top_n)
        st.success(f"Top {top_n} recommendations for user {selected_user}")
        st.dataframe(rec)
    except Exception as e:
        st.error(f"Error: {e}")

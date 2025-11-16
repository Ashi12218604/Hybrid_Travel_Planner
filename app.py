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

try:
    rating_model = joblib.load("ridge_experience_model.joblib")
    HAS_RATING_MODEL = True
except:
    HAS_RATING_MODEL = False


# ==========================================================
# 3. TF-IDF VECTORIZER (must match training)
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
# 4. RECREATE THE EXACT TRAINING FEATURE PIPELINE (34 features)
# ==========================================================

def build_reg_X(df_local):
    # numeric features
    num = df_local[['Popularity', 'NumChildren']].fillna(0).astype(float)

    # one-hot categorical features (MUST match training)
    cat = pd.get_dummies(
        df_local[['Gender', 'DestType', 'BestTimeToVisit']].fillna('Unknown').astype(str),
        dummy_na=False
    )

    # Align one-hot columns with training categories (34 consistency)
    # Determine expected columns = 34 - TF-IDF dims - 2 numeric
    tfidf_dim = dest_tfidf.shape[1]
    expected_tab_dim = 34 - tfidf_dim

    # If OHE generated fewer columns, pad
    if cat.shape[1] < expected_tab_dim - 2:
        missing = (expected_tab_dim - 2) - cat.shape[1]
        for i in range(missing):
            cat[f"_pad_{i}"] = 0

    # If too many, trim extra
    if cat.shape[1] > expected_tab_dim - 2:
        cat = cat.iloc[:, : (expected_tab_dim - 2)]

    # combine numeric + one-hot
    tab = np.hstack([num.values, cat.values])

    # destination TF-IDF vectors
    dest_vecs = []
    for did in df_local['DestinationID']:
        di = dest_id_to_idx.get(int(did), None)
        if di is None:
            dest_vecs.append(np.zeros(tfidf_dim))
        else:
            dest_vecs.append(dest_tfidf[di].toarray().ravel())
    dest_vecs = np.vstack(dest_vecs)

    # final feature matrix
    Xfull = np.hstack([tab, dest_vecs])
    return Xfull


# ==========================================================
# 5. HYBRID RECOMMENDER FUNCTION
# ==========================================================

def recommend_for_user(user_id, top_n=10, alpha=0.4, beta=0.3, gamma=0.3):

    user = df_users[df_users["UserID"] == user_id]
    if user.empty:
        raise ValueError("Invalid User ID selected.")

    gender = user["Gender"].iloc[0]
    preference = user.get("Preference", pd.Series(["Unknown"])).iloc[0]
    num_children = user.get("NumberOfChildren",
                            user.get("NumChildren", pd.Series([0]))).iloc[0]

    # Content similarity
    user_text = f"{preference} {gender}"
    user_vec = tfidf.transform([user_text])
    content_sim = cosine_similarity(user_vec, dest_tfidf).ravel()

    # Visit probability features
    pair_df = pd.DataFrame({
        "Popularity": df_dest["Popularity"].astype(float),
        "NumChildren": num_children,
        "Gender": gender,
        "DestType": df_dest["Type"].astype(str),
        "BestTimeToVisit": df_dest["BestTimeToVisit"].astype(str),
        "ContentSim": content_sim
    })
    visit_probs = visit_model.predict_proba(pair_df[["Popularity", "NumChildren", "ContentSim"]])[:, 1]

    # Rating prediction — FULL PIPELINE
    df_reg_features = pd.DataFrame({
        "Popularity": df_dest["Popularity"].astype(float),
        "NumChildren": num_children,
        "Gender": gender,
        "DestType": df_dest["Type"].astype(str),
        "BestTimeToVisit": df_dest["BestTimeToVisit"].astype(str),
        "DestinationID": df_dest["DestinationID"].astype(int)
    })

    if HAS_RATING_MODEL:
        X_for_rating = build_reg_X(df_reg_features)
        pred_rating = rating_model.predict(X_for_rating)
    else:
        pop = df_dest["Popularity"].astype(float)
        pred_rating = 1 + 4 * (pop - pop.min()) / (pop.max() - pop.min())


    def norm(x):
        x = np.array(x, dtype=float)
        return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else np.zeros_like(x)

    R = norm(pred_rating)
    V = norm(visit_probs)
    C = norm(content_sim)

    hybrid_score = alpha * R + beta * V + gamma * C

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
# 6. STREAMLIT UI
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

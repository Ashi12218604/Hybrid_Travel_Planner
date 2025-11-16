import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# 1. LOAD & STANDARDIZE DATA
# ==========================================================

df_users = pd.read_csv("Final_Updated_Expanded_Users.csv")
df_dest = pd.read_csv("Expanded_Destinations.csv")

df_users.columns = [c.strip() for c in df_users.columns]
df_dest.columns = [c.strip() for c in df_dest.columns]

# STANDARDIZE COLUMN NAMES
if "Type" in df_dest.columns:
    df_dest.rename(columns={"Type": "DestType"}, inplace=True)

if "NumberOfChildren" in df_users.columns:
    df_users.rename(columns={"NumberOfChildren": "NumChildren"}, inplace=True)

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
# 3. TF-IDF VECTORIZER
# ==========================================================

dest_texts = (
    df_dest["Name"].fillna("") + " " +
    df_dest["DestType"].fillna("") + " " +
    df_dest["State"].fillna("")
).astype(str)

user_pref_texts = df_users.get("Preference", pd.Series([""])).astype(str)

tfidf = TfidfVectorizer(max_features=500)
tfidf.fit(pd.concat([dest_texts, user_pref_texts], ignore_index=True))

dest_tfidf = tfidf.transform(dest_texts)
dest_id_to_idx = {int(did): i for i, did in enumerate(df_dest["DestinationID"])}


# ==========================================================
# 4. EXACT TRAINING PIPELINE (34 features)
# ==========================================================

def build_reg_X(df_local):
    # numeric
    num = df_local[['Popularity', 'NumChildren']].fillna(0).astype(float)

    # one-hot
    cat = pd.get_dummies(
        df_local[['Gender', 'DestType', 'BestTimeToVisit']].fillna("Unknown").astype(str),
        dummy_na=False
    )

    tfidf_dim = dest_tfidf.shape[1]
    expected_total = 34
    expected_tab_dim = expected_total - tfidf_dim

    # align one-hot columns
    while cat.shape[1] < (expected_tab_dim - 2):
        cat[f"_pad_{cat.shape[1]}"] = 0

    if cat.shape[1] > (expected_tab_dim - 2):
        cat = cat.iloc[:, : (expected_tab_dim - 2)]

    tab = np.hstack([num.values, cat.values])

    # destination vectors
    dest_vecs = []
    for did in df_local["DestinationID"]:
        idx = dest_id_to_idx.get(int(did), None)
        if idx is None:
            dest_vecs.append(np.zeros(tfidf_dim))
        else:
            dest_vecs.append(dest_tfidf[idx].toarray().ravel())

    dest_vecs = np.vstack(dest_vecs)

    return np.hstack([tab, dest_vecs])


# ==========================================================
# 5. HYBRID RECOMMENDER
# ==========================================================

def recommend_for_user(user_id, top_n=10):

    user = df_users[df_users["UserID"] == user_id]
    if user.empty:
        raise ValueError("Invalid User ID")

    gender = user["Gender"].iloc[0]
    preference = user["Preference"].iloc[0]
    num_children = user["NumChildren"].iloc[0]

    # content similarity
    user_vec = tfidf.transform([f"{preference} {gender}"])
    content_sim = cosine_similarity(user_vec, dest_tfidf).ravel()

    # visit probability
    pair_df = pd.DataFrame({
        "Popularity": df_dest["Popularity"].astype(float),
        "NumChildren": num_children,
        "Gender": gender,
        "DestType": df_dest["DestType"],
        "BestTimeToVisit": df_dest["BestTimeToVisit"],
        "ContentSim": content_sim
    })

    visit_probs = visit_model.predict_proba(pair_df[["Popularity", "NumChildren", "ContentSim"]])[:, 1]

    # rating prediction
    reg_df = pd.DataFrame({
        "Popularity": df_dest["Popularity"].astype(float),
        "NumChildren": num_children,
        "Gender": gender,
        "DestType": df_dest["DestType"].astype(str),
        "BestTimeToVisit": df_dest["BestTimeToVisit"].astype(str),
        "DestinationID": df_dest["DestinationID"].astype(int)
    })

    if HAS_RATING_MODEL:
        X_reg = build_reg_X(reg_df)
        pred_rating = rating_model.predict(X_reg)
    else:
        pop = df_dest["Popularity"].astype(float)
        pred_rating = 1 + 4 * (pop - pop.min()) / (pop.max() - pop.min())

    # hybrid score
    def norm(x):
        x = np.array(x, dtype=float)
        if x.max() == x.min():
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    R = norm(pred_rating)
    V = norm(visit_probs)
    C = norm(content_sim)

    hybrid = 0.4 * R + 0.3 * V + 0.3 * C

    out = df_dest.copy()
    out["PredRating"] = pred_rating
    out["VisitProb"] = visit_probs
    out["ContentSim"] = content_sim
    out["HybridScore"] = hybrid

    return out.sort_values("HybridScore", ascending=False).head(top_n)[[
        "DestinationID", "Name", "DestType", "State",
        "Popularity", "PredRating", "VisitProb", "ContentSim", "HybridScore"
    ]]


# ==========================================================
# 6. STREAMLIT UI
# ==========================================================

st.title("🌍 Hybrid Travel Planner Recommendation System")

user_list = df_users["UserID"].dropna().astype(int).tolist()
uid = st.selectbox("Select a User ID", user_list)

k = st.slider("Number of Recommendations", 5, 20, 10)

if st.button("Recommend"):
    try:
        res = recommend_for_user(uid, k)
        st.dataframe(res)
    except Exception as e:
        st.error(f"Error: {e}")

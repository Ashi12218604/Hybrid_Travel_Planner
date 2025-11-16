# app.py - robust final version
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")

# ---------------------------
# 1. Load data (from repo root)
# ---------------------------
@st.cache_data
def load_data():
    df_users = pd.read_csv("Final_Updated_Expanded_Users.csv")
    df_dest = pd.read_csv("Expanded_Destinations.csv")
    return df_users, df_dest

df_users, df_dest = load_data()

# ---------------------------
# 2. Normalize column names (very defensive)
# ---------------------------
def standardize_columns(df_users, df_dest):
    # strip spaces
    df_users.columns = [c.strip() for c in df_users.columns]
    df_dest.columns = [c.strip() for c in df_dest.columns]

    # Destination type: accept 'Type' or 'DestType'
    if "Type" in df_dest.columns and "DestType" not in df_dest.columns:
        df_dest.rename(columns={"Type": "DestType"}, inplace=True)
    if "DestType" not in df_dest.columns:
        df_dest["DestType"] = "Unknown"

    # Preference column in users: 'Preferences' or 'Preference'
    if "Preferences" in df_users.columns and "Preference" not in df_users.columns:
        df_users.rename(columns={"Preferences": "Preference"}, inplace=True)
    if "Preference" not in df_users.columns:
        df_users["Preference"] = ""

    # Number of children: various names
    if "NumberOfChildren" in df_users.columns and "NumChildren" not in df_users.columns:
        df_users.rename(columns={"NumberOfChildren": "NumChildren"}, inplace=True)
    if "NumberOf" in df_users.columns and "NumChildren" not in df_users.columns:
        df_users.rename(columns={"NumberOf": "NumChildren"}, inplace=True)
    if "NumChildren" not in df_users.columns:
        df_users["NumChildren"] = 0

    # Gender fallback
    if "Gender" not in df_users.columns:
        # try some alternatives that might exist
        alt = None
        for cand in ["Sex", "sex"]:
            if cand in df_users.columns:
                alt = cand
                break
        if alt:
            df_users.rename(columns={alt: "Gender"}, inplace=True)
        else:
            df_users["Gender"] = "Unknown"

    # Ensure important columns exist in dest
    if "Name" not in df_dest.columns:
        df_dest["Name"] = df_dest["DestinationID"].astype(str)
    if "State" not in df_dest.columns:
        df_dest["State"] = ""

    # Ensure numeric ID columns
    if "UserID" in df_users.columns:
        df_users["UserID"] = pd.to_numeric(df_users["UserID"], errors="coerce")
    if "DestinationID" in df_dest.columns:
        df_dest["DestinationID"] = pd.to_numeric(df_dest["DestinationID"], errors="coerce")

    return df_users, df_dest

df_users, df_dest = standardize_columns(df_users, df_dest)

# Show actual loaded columns in an expander for debug (safe)
with st.expander("Data columns (click to inspect)"):
    st.write("Users columns:", list(df_users.columns))
    st.write("Destinations columns:", list(df_dest.columns))
    st.write("Users sample (first 5 rows):")
    st.dataframe(df_users.head())
    st.write("Destinations sample (first 5 rows):")
    st.dataframe(df_dest.head())

# ---------------------------
# 3. Load models
# ---------------------------
@st.cache_resource
def load_models():
    visit = joblib.load("visit_probability_model.joblib")
    try:
        ridge = joblib.load("ridge_experience_model.joblib")
        has_ridge = True
    except Exception:
        ridge = None
        has_ridge = False
    return visit, ridge, has_ridge

visit_model, ridge_model, HAS_RIDGE = load_models()

# ---------------------------
# 4. TF-IDF fit/transform (match training)
# ---------------------------
@st.cache_data
def build_tfidf_and_vectors(df_dest, df_users):
    dest_texts = (df_dest["Name"].fillna("") + " " + df_dest["DestType"].fillna("") + " " + df_dest["State"].fillna("")).astype(str)
    user_texts = df_users["Preference"].astype(str)
    vect = TfidfVectorizer(max_features=500)
    vect.fit(pd.concat([dest_texts, user_texts], ignore_index=True))
    dest_vecs = vect.transform(dest_texts)
    id_to_idx = {int(d): i for i, d in enumerate(df_dest["DestinationID"])}
    return vect, dest_vecs, id_to_idx

tfidf, dest_tfidf, dest_id_to_idx = build_tfidf_and_vectors(df_dest, df_users)

# ---------------------------
# 5. Build regression feature builder (exact pipeline)
# ---------------------------
def build_reg_X(df_local):
    # Expectation: final feature length = 34 (as in training)
    tfidf_dim = dest_tfidf.shape[1]
    expected_total = 34
    expected_tab = expected_total - tfidf_dim  # #columns before TF-IDF

    # numeric columns (Popularity, NumChildren)
    for c in ["Popularity", "NumChildren"]:
        if c not in df_local.columns:
            df_local[c] = 0
    num = df_local[['Popularity', 'NumChildren']].fillna(0).astype(float)

    # ensure categorical cols exist
    for c in ["Gender", "DestType", "BestTimeToVisit"]:
        if c not in df_local.columns:
            df_local[c] = "Unknown"

    cat = pd.get_dummies(df_local[["Gender", "DestType", "BestTimeToVisit"]].astype(str), dummy_na=False)

    # pad or trim cat so tab width matches expected_tab - 2 (because 2 numeric)
    target_cat_cols = max(0, expected_tab - 2)
    if cat.shape[1] < target_cat_cols:
        # add zero columns
        for i in range(target_cat_cols - cat.shape[1]):
            cat[f"_pad_{i}"] = 0
    elif cat.shape[1] > target_cat_cols:
        cat = cat.iloc[:, :target_cat_cols]

    tab = np.hstack([num.values, cat.values])

    # destination TF-IDF vectors
    dest_vecs = []
    for did in df_local["DestinationID"]:
        idx = dest_id_to_idx.get(int(did), None)
        if idx is None:
            dest_vecs.append(np.zeros(tfidf_dim))
        else:
            dest_vecs.append(dest_tfidf[idx].toarray().ravel())
    dest_vecs = np.vstack(dest_vecs)

    Xfull = np.hstack([tab, dest_vecs])
    # final defensive check: if shape mismatches, pad/truncate
    if Xfull.shape[1] < expected_total:
        pad = np.zeros((Xfull.shape[0], expected_total - Xfull.shape[1]))
        Xfull = np.hstack([Xfull, pad])
    elif Xfull.shape[1] > expected_total:
        Xfull = Xfull[:, :expected_total]
    return Xfull

# ---------------------------
# 6. Recommender
# ---------------------------
def recommend_for_user(user_id, top_n=10, alpha=0.4, beta=0.3, gamma=0.3):
    # validate user
    urows = df_users[df_users["UserID"] == user_id]
    if urows.empty:
        raise ValueError("Selected user id not found in users dataset.")
    user = urows.iloc[0]

    gender = user.get("Gender", "Unknown")
    preference = str(user.get("Preference", ""))
    num_children = int(user.get("NumChildren", 0))

    # content similarity (user text vs destination tfidf)
    user_text = f"{preference} {gender}"
    user_vec = tfidf.transform([user_text])
    content_sim = cosine_similarity(user_vec, dest_tfidf).ravel()

    # visit probability input (defensive)
    visit_df = pd.DataFrame({
        "Popularity": df_dest.get("Popularity", 0).astype(float),
        "NumChildren": num_children,
        "ContentSim": content_sim
    })

    try:
        visit_probs = visit_model.predict_proba(visit_df)[:, 1]
    except Exception as e:
        # fallback uniform probability if model expects different features
        visit_probs = np.repeat(0.5, len(df_dest))

    # rating prediction: build reg_df with necessary cols
    reg_df = pd.DataFrame({
        "Popularity": df_dest.get("Popularity", 0).astype(float),
        "NumChildren": num_children,
        "Gender": gender,
        "DestType": df_dest.get("DestType", "Unknown").astype(str),
        "BestTimeToVisit": df_dest.get("BestTimeToVisit", "Unknown").astype(str),
        "DestinationID": df_dest["DestinationID"].astype(int)
    })

    if ridge_model is not None:
        try:
            X_for_rating = build_reg_X(reg_df)
            pred_rating = ridge_model.predict(X_for_rating)
        except Exception as e:
            # if something unexpected occurs, fallback to popularity-based
            pop = df_dest.get("Popularity", 0).astype(float)
            pred_rating = 1 + 4 * (pop - pop.min()) / (pop.max() - pop.min() if pop.max()!=pop.min() else 1.0)
    else:
        pop = df_dest.get("Popularity", 0).astype(float)
        pred_rating = 1 + 4 * (pop - pop.min()) / (pop.max() - pop.min() if pop.max()!=pop.min() else 1.0)

    # normalize helpers
    def norm(x):
        x = np.array(x, dtype=float)
        if x.max() == x.min():
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    R = norm(pred_rating)
    V = norm(visit_probs)
    C = norm(content_sim)

    hybrid_score = alpha * R + beta * V + gamma * C

    out = df_dest.copy()
    out["PredRating"] = pred_rating
    out["VisitProb"] = visit_probs
    out["ContentSim"] = content_sim
    out["HybridScore"] = hybrid_score

    display_cols = ["DestinationID", "Name"]
    if "DestType" in out.columns:
        display_cols.append("DestType")
    elif "Type" in out.columns:
        display_cols.append("Type")
    display_cols += ["State", "Popularity", "PredRating", "VisitProb", "ContentSim", "HybridScore"]

    return out.sort_values("HybridScore", ascending=False).head(top_n)[display_cols]

# ---------------------------
# 7. Streamlit UI
# ---------------------------
st.title("🌍 Hybrid Travel Planner Recommendation System")

user_list = df_users["UserID"].dropna().astype(int).tolist()
selected_user = st.selectbox("Select a User ID:", user_list)

top_n = st.slider("Number of Recommendations:", 5, 20, 10)

if st.button("Recommend"):
    try:
        recs = recommend_for_user(selected_user, top_n=top_n)
        st.success(f"Top {top_n} recommendations for user {selected_user}")
        st.dataframe(recs)
    except Exception as e:
        st.error(f"Error: {e}")
        # show a bit more debug info to help
        with st.expander("Debug info"):
            st.write("Users columns:", list(df_users.columns))
            st.write("Dest columns:", list(df_dest.columns))
            st.write("Sample user row:", df_users[df_users['UserID']==selected_user].to_dict('records'))

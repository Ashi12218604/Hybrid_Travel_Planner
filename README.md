
# **Hybrid Travel Planner – Machine Learning Recommendation System**

### **Live Demo:** [https://hybridtravelplanner.streamlit.app/](https://hybridtravelplanner.streamlit.app/)

---

## **1. Overview**

This project implements a **Hybrid Travel Recommendation System** that generates personalized destination suggestions for users by combining:

* **Machine Learning models**
* **Natural Language Processing**
* **User preferences & demographics**
* **Destination metadata**

The system predicts:

1. How much a user will **like** a destination (regression)
2. How likely they are to **visit** the destination (classification)
3. How strongly a destination’s description matches the user’s **preferences** (NLP similarity)

These signals are combined using a weighted hybrid scoring framework and deployed as an interactive **Streamlit web app**.

---

## **2. Features**

### **Hybrid Recommendation Engine**

* **Ridge Regression** → Predicts user experience rating
* **Random Forest Classifier** → Predicts visit probability
* **TF-IDF + Cosine Similarity** → Measures alignment between user preference text and destination descriptions
* **HybridScore** combines all three for final ranking

### **Interactive Web Interface**

* Built with **Streamlit**
* Select any user
* Choose number of recommendations
* View top destinations with predicted rating, visit probability, similarity score, and hybrid ranking

### **End-to-End ML Pipeline**

* Dataset cleaning & preprocessing
* Feature engineering
* Model training, evaluation, saving
* Hybrid score computation
* Deployment

---

## **3. Datasets**

This project uses multiple CSV datasets:

| File                                     | Description                                                               |
| ---------------------------------------- | ------------------------------------------------------------------------- |
| `Final_Updated_Expanded_Users.csv`       | User demographics, preferences, and family information                    |
| `Expanded_Destinations.csv`              | Destination metadata including type, popularity, and seasonal suitability |
| `Final_Updated_Expanded_UserHistory.csv` | Travel history & experience ratings                                       |
| `Final_Updated_Expanded_Reviews.csv`     | Text reviews and ratings of destinations                                  |

Combined, they form a rich feature space for modeling user behavior.

---

## **4. Machine Learning Components**

### **4.1 Regression Model – Ridge Regression**

Used to estimate **ExperienceRating** for user–destination pairs.

Includes concepts from:

* Regularization (L2)
* Coefficient estimation
* Cross-validation
* Handling multicollinearity

The regression model was trained on a **34-feature input vector**, combining:

* Numeric features
* One-hot encoded categorical features
* Destination TF-IDF vectors

### **4.2 Classification Model – Random Forest**

Used to predict the **Visit Probability** of a user.

Covers:

* Decision trees
* Ensemble learning
* Bagging
* Impurity measures (Gini)
* Feature importance

### **4.3 NLP Model – TF-IDF Similarity**

Used to compute:

* User preference → Destination similarity
* Cosine similarity between text embeddings

---

## **5. Hybrid Recommendation Formula**

The final score used to rank destinations:

```
HybridScore = 0.4 × PredictedRating  
            + 0.3 × VisitProbability  
            + 0.3 × ContentSimilarity
```

Each component is normalized to maintain equal scale.

---

## **6. Project Structure**

```
📁 Hybrid Travel Planner
│
├── app.py
├── ridge_experience_model.joblib
├── visit_probability_model.joblib
│
├── Final_Updated_Expanded_Users.csv
├── Final_Updated_Expanded_UserHistory.csv
├── Final_Updated_Expanded_Reviews.csv
├── Expanded_Destinations.csv
│
└── README.md
```

---

## **7. Deployment**

The application is deployed using **Streamlit Cloud**.

You can access the live application here:
🔗 **[https://hybridtravelplanner.streamlit.app/](https://hybridtravelplanner.streamlit.app/)**

To run it locally:

### **Install dependencies**

```
pip install -r requirements.txt
```

### **Run the app**

```
streamlit run app.py
```

---

## **8. Visualizations**

The notebook includes key graphs:

* Distribution of destination popularity
* User preference frequency
* Random Forest feature importance
* Similarity heatmaps
* Predicted rating distributions

These help validate model behavior and explain the system.

---

## **9. Conclusion**

This project demonstrates how multiple ML paradigms—regression, classification, NLP, and ensemble learning—can be integrated to build a real-world hybrid recommender system.
The system is modular, extensible, and reflects industry practices in travel recommendation engines.

It serves as a strong applied machine learning project showcasing:

* dataset handling
* supervised model training
* NLP engineering
* end-to-end pipeline building
* web-based deployment

---



import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import io
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, confusion_matrix

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

# ================= THEME =================
st.markdown("""
<style>
body, .stApp {background-color: #0B0B14; color: #FFFFFF;}
h1,h2,h3 {color:#7C5CFF;}
.card {background:#0F0F1A; border:1px solid #7C5CFF; padding:20px; border-radius:15px; margin-bottom:20px;}
.metric {background:#0F0F1A; color:#7C5CFF; padding:15px; border-radius:12px; text-align:center; margin:10px;}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<h1>🤖 AutoML Intelligence Platform</h1>", unsafe_allow_html=True)

# ================= FILE =================
file = st.file_uploader("Upload Dataset", type=["csv","xlsx"])

# ================= CONFIG =================
problem_mode = st.radio("Mode", ["Auto Detect","Classification","Regression"])
test_size = st.slider("Test Size %", 10, 40, 20)

# ================= FUNCTIONS =================
def detect_task(y):
    return "classification" if y.dtype=="object" or y.nunique()<20 else "regression"

def preprocess(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(exclude=np.number).columns

    # numeric
    if len(num_cols):
        num_data = SimpleImputer(strategy="mean").fit_transform(X[num_cols])
        X[num_cols] = StandardScaler().fit_transform(num_data)

    # categorical
    if len(cat_cols):
        cat_data = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])
        enc = OneHotEncoder(drop="first", sparse=False)
        cat_encoded = enc.fit_transform(cat_data)

        cat_df = pd.DataFrame(cat_encoded, columns=enc.get_feature_names_out(cat_cols))
        X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True), cat_df], axis=1)

    return X, y

# ================= MAIN =================
if file:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    X, y = preprocess(df, target)

    task = detect_task(y) if problem_mode=="Auto Detect" else problem_mode.lower()
    st.success(f"Detected Task: {task.upper()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    if st.button("🚀 Run AutoML"):

        with st.spinner("Training Models..."):

            results = []
            best_score = -999
            best_model = None
            best_name = ""

            if task=="classification":
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=500),
                    "Random Forest": RandomForestClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier(),
                    "KNN": KNeighborsClassifier(),
                    "Decision Tree": DecisionTreeClassifier()
                }
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Ridge": Ridge(),
                    "KNN": KNeighborsRegressor(),
                    "Decision Tree": DecisionTreeRegressor()
                }

            # TRAIN LOOP
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                if task=="classification":
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="weighted")
                    score = f1
                    results.append([name, acc, f1])
                else:
                    r2 = r2_score(y_test, preds)
                    mae = mean_absolute_error(y_test, preds)
                    score = r2
                    results.append([name, r2, mae])

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = name

        # ================= RESULTS =================
        st.success(f"🏆 Best Model: {best_name} ({round(best_score,4)})")

        results_df = pd.DataFrame(results, columns=["Model","Metric1","Metric2"])
        st.dataframe(results_df)

        # ================= FEATURE IMPORTANCE =================
        if hasattr(best_model, "feature_importances_"):
            fi = pd.DataFrame({
                "Feature": X.columns,
                "Importance": best_model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            fig = px.bar(fi, x="Feature", y="Importance")
            st.plotly_chart(fig)

        # ================= CONFUSION MATRIX =================
        if task=="classification":
            cm = confusion_matrix(y_test, best_model.predict(X_test))
            fig = px.imshow(cm, text_auto=True)
            st.plotly_chart(fig)

        # ================= DOWNLOAD =================
        preds = best_model.predict(X_test)
        out = X_test.copy()
        out["Prediction"] = preds

        st.subheader("Predictions")
        st.dataframe(out.head())

        st.download_button("Download CSV", out.to_csv(index=False), "predictions.csv")

        # ================= DOWNLOAD MODEL =================
        model_bytes = pickle.dumps(best_model)
        st.download_button("Download Model (.pkl)", model_bytes, "model.pkl")

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
import numpy as np
import io

# ================= PAGE STYLING =================
st.markdown("""
<style>
body, .stApp, .block-container {background-color: #0B0B14; color: #FFFFFF; font-family: 'Inter','Poppins',sans-serif;}
h1,h2,h3,h4,h5,h6 {color: #7C5CFF !important;}
.card, .section-card {background-color: #0F0F1A; border: 1px solid #7C5CFF; border-radius: 18px; padding: 22px; box-shadow: 0 0 25px rgba(124,92,255,0.4); margin-bottom: 22px;}
.hero-card {background:#0B0B14;color:#7C5CFF;padding:42px;border-radius:26px;box-shadow:0 0 40px rgba(124,92,255,0.9);text-align:center;}
div[data-testid="stFileUploader"] > div {background: linear-gradient(135deg,rgba(124,92,255,0.15),rgba(124,92,255,0.05)) !important; border:1px solid #7C5CFF !important; border-radius:18px !important; padding:26px !important; box-shadow: inset 0 0 20px rgba(124,92,255,0.2), 0 0 15px rgba(124,92,255,0.3);}
div[data-testid="stFileUploader"] span, div[data-testid="stFileUploader"] p {color:#FFFFFF !important; font-weight:600;}
div[data-testid="stFileUploader"] button {background:linear-gradient(135deg,#7C5CFF,#9A84FF) !important; color:#FFFFFF !important; border-radius:12px; box-shadow:0 0 15px rgba(124,92,255,0.6); font-weight:600;}
.css-14xtw13 input:checked + div {background: linear-gradient(135deg,#7C5CFF,#9A84FF) !important; border-color:#7C5CFF !important;}
.css-14xtw13 div {color:#FFFFFF !important;}
.css-1kyxreq .stSlider > div > div:nth-child(1) {background: linear-gradient(135deg,#7C5CFF,#9A84FF) !important;}
.css-1kyxreq .stSlider > div > div:nth-child(2) {background-color: #7C5CFF !important;}
.css-1kyxreq .stSlider > div > div:nth-child(3) {background-color: #7C5CFF !important;}
div.stButton > button, div.stDownloadButton > button {background:linear-gradient(135deg,#7C5CFF,#9A84FF) !important; color:#FFFFFF !important; border-radius:14px; padding:0.7em 1.4em !important; font-weight:600; box-shadow:0 0 25px rgba(124,92,255,0.6);}
div.stButton > button:hover {transform:translateY(-2px); box-shadow:0 0 40px rgba(124,92,255,0.9);}
.metric {background:#0F0F1A;color:#7C5CFF;padding:22px;border-radius:16px;text-align:center;font-weight:700;box-shadow:0 0 30px rgba(124,92,255,0.8);}
.stDataFrame, table {background-color:#0F0F1A; border:1px solid #7C5CFF; color:#FFFFFF;}
a {color:#7C5CFF;}
a:hover {text-shadow:0 0 12px rgba(124,92,255,0.9);}
</style>
""", unsafe_allow_html=True)

# ================= HERO =================
st.markdown("""
<div class="hero-card">
<h1>🤖 AutoML Intelligence Platform</h1>
<p style="font-size:18px; opacity:0.9;">Automated Machine Learning for Classification & Regression</p>
</div>
""", unsafe_allow_html=True)

# ================= FILE UPLOADER =================
file = st.file_uploader("📂 Upload Dataset (CSV / Excel)", type=["csv","xlsx"])

# ================= PROBLEM CONFIG =================
st.subheader("🧠 Problem Configuration")
problem_mode = st.radio("Mode", ["Auto Detect", "Classification", "Regression"])

# ================= MODEL CONFIG =================
st.subheader("🔧 Model Configuration")
test_size = st.slider("Test Size (%)", 10, 40, 20)
n_neighbors = st.slider("KNN Neighbors", 1, 20, 5)
max_depth = st.slider("Tree Depth", 2, 20, 5)
n_estimators = st.slider("Estimators", 50, 500, 100, step=50)

# ================= EVALUATION OPTIONS =================
st.subheader("📊 Evaluation Options")
show_feature_importance = st.checkbox("Feature Importance", True)
show_confusion = st.checkbox("Confusion Matrix", True)

# ================= FUNCTIONS =================
def detect_task(y):
    return "classification" if y.dtype=="object" or y.nunique()<=20 else "regression"

def preprocess(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    num_cols = X.select_dtypes(include=["int64","float64"]).columns
    cat_cols = X.select_dtypes(include=["object","category"]).columns
    if len(num_cols):
        X[num_cols] = StandardScaler().fit_transform(
            SimpleImputer(strategy="mean").fit_transform(X[num_cols])
        )
    if len(cat_cols):
        enc = OneHotEncoder(drop="first", sparse_output=False)
        X_cat = enc.fit_transform(
            SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])
        )
        X_cat_df = pd.DataFrame(X_cat, columns=enc.get_feature_names_out(cat_cols), index=X.index)
        X = pd.concat([X.drop(columns=cat_cols), X_cat_df], axis=1)
    return X, y

# ================= MAIN =================
if file:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    # ✅ Auto-select target for obesity dataset
    default_target = "NObeyesdad" if "NObeyesdad" in df.columns else df.columns[0]
    target = st.selectbox("🎯 Select Target Column", df.columns, index=list(df.columns).index(default_target))

    X, y = preprocess(df, target)
    task = detect_task(y) if problem_mode=="Auto Detect" else problem_mode.lower()
    st.info(f"Detected Task: {task.upper()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    # ================= TABS =================
    tab1, tab2, tab3 = st.tabs(["📊 EDA","🧠 Models","💾 Downloads"])

    # ======== TAB 1: EDA ========
    with tab1:
        st.markdown("<div class='card'>Dataset Preview</div>", unsafe_allow_html=True)
        st.dataframe(df.head())

        st.markdown("<div class='card'>Dataset Info</div>", unsafe_allow_html=True)
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.markdown(f"<pre style='color:white'>{buffer.getvalue()}</pre>", unsafe_allow_html=True)

        st.markdown("<div class='card'>Summary Statistics</div>", unsafe_allow_html=True)
        st.dataframe(df.describe())

        st.markdown("<div class='card'>Correlation Heatmap</div>", unsafe_allow_html=True)
        corr = df.select_dtypes(include=['float64', 'int64']).corr()
        fig = px.imshow(corr, text_auto=True,
                        color_continuous_scale=[(0,'rgba(124,92,255,0.1)'), (1,'rgba(124,92,255,0.7)')])
        fig.update_layout(plot_bgcolor='#0B0B14', paper_bgcolor='#0B0B14', font_color='#FFFFFF')
        st.plotly_chart(fig, use_container_width=True)

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            st.markdown(f"<div class='card'>Histogram: {col}</div>", unsafe_allow_html=True)
            fig = px.histogram(df, x=col)
            fig.update_traces(marker=dict(color='rgba(124,92,255,0.3)', line=dict(color='#7C5CFF', width=3)))
            fig.update_layout(plot_bgcolor='#0B0B14', paper_bgcolor='#0B0B14', font_color='#FFFFFF')
            st.plotly_chart(fig, use_container_width=True)

    # ======== TAB 2: MODELS ========
    with tab2:
        if st.button("🚀 Run AutoML"):
            results = []
            metrics_list = []

            if task=="classification":
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=500),
                    "Random Forest": RandomForestClassifier(n_estimators=n_estimators),
                    "Gradient Boosting": GradientBoostingClassifier(n_estimators=n_estimators),
                    "KNN Classifier": KNeighborsClassifier(n_neighbors=n_neighbors),
                    "Decision Tree": DecisionTreeClassifier(max_depth=max_depth)
                }
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(n_estimators=n_estimators),
                    "Gradient Boosting": GradientBoostingRegressor(n_estimators=n_estimators),
                    "Ridge Regression": Ridge(),
                    "KNN Regressor": KNeighborsRegressor(n_neighbors=n_neighbors),
                    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=max_depth)
                }

            best_score, best_model, best_name = -np.inf, None, None

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                if task=="classification":
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="weighted")
                    score = f1
                    metrics_list.append((name, round(acc,4), round(f1,4)))
                else:
                    r2 = r2_score(y_test, preds)
                    mae = mean_absolute_error(y_test, preds)
                    score = r2
                    metrics_list.append((name, round(r2,4), round(mae,4)))

                if score > best_score:
                    best_score, best_model, best_name = score, model, name

            st.success(f"Best Model: {best_name} ({round(best_score,4)})")

            if task=="classification":
                st.markdown("### 📈 Classification Metrics (Accuracy / F1 Weighted)")
                for n, acc, f1 in metrics_list:
                    st.markdown(f"<div class='metric'>{n}<br>Accuracy: {acc} | F1: {f1}</div>", unsafe_allow_html=True)
            else:
                st.markdown("### 📈 Regression Metrics (R² / MAE)")
                for n, r2, mae in metrics_list:
                    st.markdown(f"<div class='metric'>{n}<br>R²: {r2} | MAE: {mae}</div>", unsafe_allow_html=True)

            if show_feature_importance and hasattr(best_model, "feature_importances_"):
                fi = pd.DataFrame({"Feature": X.columns, "Importance": best_model.feature_importances_}).sort_values(by="Importance", ascending=False)
                fig = px.bar(fi, x="Feature", y="Importance", text="Importance")
                fig.update_traces(marker=dict(color='rgba(124,92,255,0.3)', line=dict(color='#7C5CFF', width=3)), textposition='outside')
                fig.update_layout(plot_bgcolor='#0B0B14', paper_bgcolor='#0B0B14', font_color='#FFFFFF', xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            if task=="classification" and show_confusion:
                cm = confusion_matrix(y_test, best_model.predict(X_test))
                st.markdown("<div class='card'>Confusion Matrix</div>", unsafe_allow_html=True)
                fig = px.imshow(cm, text_auto=True, color_continuous_scale=[(0,'rgba(124,92,255,0.1)'), (1,'rgba(124,92,255,0.7)')], aspect="auto")
                fig.update_layout(plot_bgcolor='#0B0B14', paper_bgcolor='#0B0B14', font_color='#FFFFFF', xaxis_title="Predicted", yaxis_title="Actual")
                st.plotly_chart(fig, use_container_width=True)

    # ======== TAB 3: DOWNLOADS ========
    with tab3:
        if 'best_model' in locals():
            preds = best_model.predict(X_test)
            out = X_test.copy()
            out["Prediction"] = preds
            st.dataframe(out.head())

            st.markdown("<div class='card'>Download Predictions</div>", unsafe_allow_html=True)
            file_format = st.selectbox("Select File Format", ["CSV", "Excel", "JSON"])
            if file_format=="CSV":
                st.download_button("📥 Download CSV", out.to_csv(index=False), "predictions.csv", mime="text/csv")
            elif file_format=="Excel":
                buffer = io.BytesIO()
                out.to_excel(buffer, index=False, engine='openpyxl')
                st.download_button("📥 Download Excel", buffer, "predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.download_button("📥 Download JSON", out.to_json(orient="records"), "predictions.json", mime="application/json")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# ---------------- CSS Styling ---------------- #

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

.hero-title{
font-size:80px;
font-weight:900;
text-align:center;
background: linear-gradient(90deg,#00dbde,#fc00ff);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin-top:20px;
margin-bottom:10px;
}

.hero-subtitle{
font-size:28px;
text-align:center;
color:#e0e0e0;
margin-bottom:40px;
}

.card{
background: rgba(255,255,255,0.08);
backdrop-filter: blur(10px);
border-radius:15px;
padding:25px;
box-shadow:0 8px 32px rgba(0,0,0,0.3);
margin-bottom:25px;
}

.section-title{
font-size:30px;
font-weight:700;
margin-bottom:10px;
color:#00e5ff;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Title ---------------- #

st.markdown('<div class="hero-title">Employee Attrition Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Machine Learning Based HR Analytics Dashboard</div>', unsafe_allow_html=True)

# ---------------- Dataset Format ---------------- #

st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">Required Dataset Format</div>', unsafe_allow_html=True)

st.write("""
Your CSV dataset must contain the following columns:

Age  
MonthlyIncome  
JobSatisfaction  
YearsAtCompany  
WorkLifeBalance  
OverTime  
Attrition
""")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Upload Dataset ---------------- #

st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">Upload Employee Dataset</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- If dataset uploaded ---------------- #

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    required_columns = [
        "Age",
        "MonthlyIncome",
        "JobSatisfaction",
        "YearsAtCompany",
        "WorkLifeBalance",
        "OverTime",
        "Attrition"
    ]

    if not all(col in data.columns for col in required_columns):
        st.error("Dataset format incorrect. Please upload dataset with required columns.")
        st.stop()

    # ---------------- Model Training ---------------- #

    X = data.drop("Attrition", axis=1)
    y = data["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # ---------------- Performance Metrics ---------------- #

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy,2))
    col2.metric("Precision", round(precision,2))
    col3.metric("Recall", round(recall,2))

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- ROC Curve ---------------- #

    y_prob = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    st.subheader("ROC Curve")

    fig = plt.figure()

    plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    st.pyplot(fig)

    # ---------------- Prediction ---------------- #

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Predict Employee Attrition</div>', unsafe_allow_html=True)

    age = st.number_input("Age", 18, 60)

    income = st.number_input("Monthly Income", 2000, 20000)

    satisfaction = st.slider("Job Satisfaction", 1, 4)

    years = st.slider("Years At Company", 0, 20)

    worklife = st.slider("Work Life Balance", 1, 4)

    overtime = st.selectbox("OverTime (0=No, 1=Yes)", [0,1])

    input_df = pd.DataFrame({
        "Age":[age],
        "MonthlyIncome":[income],
        "JobSatisfaction":[satisfaction],
        "YearsAtCompany":[years],
        "WorkLifeBalance":[worklife],
        "OverTime":[overtime]
    })

    if st.button("Predict Attrition"):

        prediction = model.predict(input_df)

        if prediction[0] == 1:
            st.error("Employee likely to leave the company")
        else:
            st.success("Employee likely to stay in the company")

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- KMeans Clustering ---------------- #

    st.subheader("Employee Clustering using K-Means")

    k = st.slider("Number of Clusters", 2, 6, 3)

    kmeans = KMeans(n_clusters=k)

    clusters = kmeans.fit_predict(X)

    fig2 = plt.figure()

    plt.scatter(X["Age"], X["MonthlyIncome"], c=clusters)

    plt.xlabel("Age")
    plt.ylabel("Monthly Income")
    plt.title("KMeans Clusters")

    st.pyplot(fig2)

    # ---------------- DBSCAN ---------------- #

    st.subheader("Employee Clustering using DBSCAN")

    eps = st.slider("EPS Value", 0.1, 10.0, 1.0)

    min_samples = st.slider("Min Samples", 2, 10, 5)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    db_clusters = dbscan.fit_predict(X)

    fig3 = plt.figure()

    plt.scatter(X["Age"], X["MonthlyIncome"], c=db_clusters)

    plt.xlabel("Age")
    plt.ylabel("Monthly Income")
    plt.title("DBSCAN Clusters")

    st.pyplot(fig3)
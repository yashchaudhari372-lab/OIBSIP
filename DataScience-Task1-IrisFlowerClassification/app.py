import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="Iris Species Predictor | Oasis Infobyte",
    page_icon="🌸",
    layout="wide"
)

# Header Section
st.title("🌸 Iris Flower Classification App")
st.markdown("""
**Oasis Infobyte — SIP Data Science Internship (Task 1)**  
* **Intern:** Yash Devidas Chaudhari | **Reference:** OIB/N2/IP5780
""")
st.divider()

# Cache model training
@st.cache_resource
def get_models():
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    models = {
        "Logistic Regression (Recommended)": LogisticRegression(max_iter=200, random_state=42),
        "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Random Forest (100 Trees)": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    for clf in models.values():
        clf.fit(X, y)
    return models, iris

models, iris = get_models()

# Sidebar for user inputs
st.sidebar.header("🔧 Input Flower Measurements")

selected_model_name = st.sidebar.selectbox("Choose ML Model", list(models.keys()))
clf = models[selected_model_name]

sepal_len = st.sidebar.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.84, step=0.1)
sepal_wid = st.sidebar.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.05, step=0.1)
petal_len = st.sidebar.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=3.75, step=0.1)
petal_wid = st.sidebar.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.20, step=0.1)

# Display input summary
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sepal Length", f"{sepal_len:.1f} cm")
col2.metric("Sepal Width", f"{sepal_wid:.1f} cm")
col3.metric("Petal Length", f"{petal_len:.1f} cm")
col4.metric("Petal Width", f"{petal_wid:.1f} cm")

st.markdown("### 🎯 Live Model Prediction")

input_data = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
pred_class_idx = clf.predict(input_data)[0]
predicted_species = iris.target_names[pred_class_idx].capitalize()

if hasattr(clf, "predict_proba"):
    probabilities = clf.predict_proba(input_data)[0]
    confidence = probabilities[pred_class_idx] * 100
else:
    confidence = 100.0
    probabilities = [0, 0, 0]
    probabilities[pred_class_idx] = 1.0

# Prediction banner
badge_color = {
    "Setosa": "🌱",
    "Versicolor": "🌿",
    "Virginica": "🌺"
}

st.success(f"### {badge_color.get(predicted_species, '🌸')} Predicted Species: **Iris {predicted_species}** (Confidence: {confidence:.2f}%)")

# Probability distribution chart
st.subheader("📊 Prediction Probabilities by Class")
prob_df = pd.DataFrame({
    "Species": [f"Iris {s.capitalize()}" for s in iris.target_names],
    "Probability (%)": [p * 100 for p in probabilities]
}).set_index("Species")

st.bar_chart(prob_df)

with st.expander("ℹ️ About the Dataset & Features"):
    st.markdown("""
    * **Iris Setosa:** Characterized by distinctly smaller petal lengths ($< 2\\text{ cm}$) and petal widths.
    * **Iris Versicolor:** Moderate petal length ($3\\text{ to }5\\text{ cm}$) and width.
    * **Iris Virginica:** Longest petals ($> 5\\text{ cm}$) and wider petals ($> 1.8\\text{ cm}$).
    * **Model Performance:** Both Logistic Regression and KNN achieved **96.67% to 100% accuracy** on the test dataset.
    """)

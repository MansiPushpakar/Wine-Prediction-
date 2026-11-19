import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Wine Class Prediction", page_icon="🍇", layout="wide")

@st.cache_resource
def train_model():
    dataset = load_wine()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    return model, X, dataset.target_names, score

model, X, class_names, score = train_model()
st.title("🍇 Wine Class Prediction")
st.caption("Machine-learning demonstration using the scikit-learn Wine dataset. Predicts one of three cultivar classes, not wine quality or price.")
left, right = st.columns([1.15, 1])
with left:
    st.subheader("Enter chemical measurements")
    st.write("Defaults are dataset medians. Adjust measurements, then select Predict.")
    with st.form("wine_form"):
        vals = {}
        for name in X.columns:
            vals[name] = st.number_input(name.replace('_', ' ').title(), value=float(X[name].median()), format="%.4f", help=f"Observed dataset range: {X[name].min():.2f} to {X[name].max():.2f}")
        submitted = st.form_submit_button("Predict wine class", type="primary", use_container_width=True)
with right:
    st.subheader("Model information")
    st.metric("Held-out test accuracy", f"{score:.1%}")
    st.metric("Training dataset", f"{len(X)} samples")
    st.write("Algorithm: Random Forest (200 trees).")
    st.write("The built-in Wine dataset contains 13 chemical measurements for each sample.")
    if submitted:
        sample = pd.DataFrame([vals], columns=X.columns)
        pred = int(model.predict(sample)[0])
        probabilities = model.predict_proba(sample)[0]
        st.success(f"Predicted class: **{class_names[pred]}** (class {pred + 1})")
        chart = pd.DataFrame({"Class": [f"{name} (class {i+1})" for i, name in enumerate(class_names)], "Probability": probabilities}).set_index("Class")
        st.bar_chart(chart)
        st.caption("Probabilities describe this model's output, not guaranteed real-world certainty.")
    else:
        st.info("Fill the form and click Predict to see a result.")
with st.expander("About the data and limitations"):
    st.write("Source: scikit-learn built-in Wine recognition dataset (178 samples, 3 classes). This educational app predicts cultivar categories. It does not estimate wine quality, taste, price, or safety. Model metrics may differ with other datasets.")

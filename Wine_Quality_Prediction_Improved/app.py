import streamlit as st
import pandas as pd

from model import (
    FEATURE_COLUMNS,
    load_wine_data,
    train_model,
    predict_quality,
    evaluate_model,
)

st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🍷 Wine Quality Prediction")
st.caption(
    "Predicts whether a red wine sample is classified as good quality "
    "using its physicochemical measurements."
)

st.info(
    "This educational model maps sensory quality scores to two classes: "
    "Good (quality ≥ 7) and Not good (quality < 7). It is not a laboratory "
    "test or a guarantee of real-world wine quality."
)

@st.cache_data(show_spinner="Loading the red wine dataset...")
def get_dataset():
    return load_wine_data()

@st.cache_resource(show_spinner="Training the Random Forest model...")
def get_model(data):
    return train_model(data)

try:
    data = get_dataset()
    model, X_test, y_test = get_model(data)
    metrics = evaluate_model(model, X_test, y_test)
except Exception as exc:
    st.error("The application could not load the dataset or train the model.")
    st.write("Check your internet connection and try refreshing the app.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

metric_cols = st.columns(4)
metric_cols[0].metric("Dataset records", f"{len(data):,}")
metric_cols[1].metric("Input features", str(len(FEATURE_COLUMNS)))
metric_cols[2].metric("Test accuracy", f"{metrics['accuracy']:.1%}")
metric_cols[3].metric("F1 score", f"{metrics['f1']:.1%}")

st.divider()
left, right = st.columns([1.25, 0.9], gap="large")

with left:
    st.subheader("Enter wine measurements")
    st.write("Use the typical dataset range shown under each measurement.")
    medians = data[FEATURE_COLUMNS].median()
    ranges = data[FEATURE_COLUMNS].agg(["min", "max"])

    with st.form("wine_prediction_form"):
        values = {}
        for feature in FEATURE_COLUMNS:
            minimum = float(ranges.loc["min", feature])
            maximum = float(ranges.loc["max", feature])
            default = float(medians[feature])
            values[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=max((maximum - minimum) / 100, 0.0001),
                format="%.4f",
                help=f"Observed range in this dataset: {minimum:.4f} to {maximum:.4f}",
            )
        submitted = st.form_submit_button(
            "Predict wine quality",
            type="primary",
            use_container_width=True,
        )

with right:
    st.subheader("About this model")
    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Target rule:** quality ≥ 7 → Good (1); otherwise Not good (0)")
    st.write("**Split:** 80% training / 20% testing")
    st.write("**Split seed:** 3, matching the original notebook")
    st.write("**Input:** 11 physicochemical measurements")
    st.caption(
        "The reported metrics are calculated on a held-out test set. "
        "They are not training accuracy."
    )

if submitted:
    try:
        result = predict_quality(model, values)
        st.divider()
        st.subheader("Prediction result")
        if result["label"] == 1:
            st.success("Predicted class: GOOD QUALITY")
        else:
            st.warning("Predicted class: NOT GOOD QUALITY")

        c1, c2 = st.columns(2)
        c1.metric("Good-quality probability", f"{result['good_probability']:.1%}")
        c2.metric("Not-good probability", f"{result['not_good_probability']:.1%}")

        probability_frame = pd.DataFrame(
            {
                "Class": ["Not good quality", "Good quality"],
                "Model probability": [
                    result["not_good_probability"],
                    result["good_probability"],
                ],
            }
        ).set_index("Class")
        st.bar_chart(probability_frame, horizontal=True)
        st.caption(
            "Probabilities are model estimates, not calibrated guarantees. "
            "A prediction of good quality means the model estimates the sample "
            "meets the notebook's quality ≥ 7 threshold."
        )
    except ValueError as exc:
        st.error(f"Please check the entered measurements: {exc}")

with st.expander("Model evaluation details"):
    st.write("Held-out test-set metrics")
    st.json({key: round(value, 4) for key, value in metrics.items()})
    st.write("Dataset class distribution after applying the original threshold")
    counts = data["quality"].ge(7).map({True: "Good (1)", False: "Not good (0)"}).value_counts()
    st.bar_chart(counts)

with st.expander("Dataset, attribution and limitations"):
    st.write(
        "The app uses the red-wine portion of the UCI Wine Quality dataset. "
        "The source contains 1,599 red-wine records, 11 physicochemical input "
        "variables and an expert sensory quality score."
    )
    st.markdown(
        "Dataset: [UCI Machine Learning Repository – Wine Quality]"
        "(https://doi.org/10.24432/C56S3T)"
    )
    st.write(
        "The original notebook converts the numeric quality score into a binary "
        "label using quality ≥ 7. This app preserves that target definition. "
        "The binary output is not the original 0–10 score."
    )
    st.write(
        "The model is trained when the app starts and cached for the session. "
        "If the public dataset host is unavailable, place `winequality-red.csv` "
        "in the project root; the app will use that local copy."
    )

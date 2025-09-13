import os, json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Math Score Predictor", page_icon="📘", layout="centered")
st.title("Student Math Score Predictor")
st.write("Enter the student attributes to estimate their **Math score**.")

# ---------- Locate model & schema ----------
MODEL_PATH = os.path.join("models", "student_model_linear.joblib")
SCHEMA_PATH = os.path.join("models", "schema.json")

def require(path, kind):
    if not os.path.exists(path):
        st.error(f"{kind} file missing: `{path}`. Make sure it's uploaded to your repo.")
        st.stop()

require(MODEL_PATH, "Model")
require(SCHEMA_PATH, "Schema")

@st.cache_resource
def load_model_and_schema():
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    model = joblib.load(MODEL_PATH)
    return model, schema

model, schema = load_model_and_schema()

cat_schema = schema.get("categorical", {})
num_schema = schema.get("numeric_ranges", {})
feature_order = schema.get("feature_order", [])
target_name = schema.get("target", "math_score")

# ---------- Build form ----------
with st.form("student_form"):
    inputs = {}
    # Categorical inputs (dropdowns from schema)
    for c, options in cat_schema.items():
        if not options:
            # if schema accidentally empty, fall back to text input
            inputs[c] = st.text_input(c.replace("_", " ").title())
        else:
            # Use the first option as default
            default_idx = 0
            inputs[c] = st.selectbox(c.replace("_", " ").title(), options, index=default_idx)

    # Numeric inputs (use median as default)
    for c, rng in num_schema.items():
        mn, mx = rng.get("min", 0.0), rng.get("max", 100.0)
        med = rng.get("median", (mn + mx) / 2)
        # Guard against equal min/max
        if mn == mx:
            mx = mn + 1.0
        step = max(1.0, (mx - mn) / 100.0)
        inputs[c] = st.number_input(c.replace("_", " ").title(), min_value=float(mn), max_value=float(mx), value=float(med), step=float(step))

    submitted = st.form_submit_button("Predict")

# ---------- Predict ----------
if submitted:
    # Keep the exact training feature order
    row = {k: inputs.get(k) for k in feature_order}
    X = pd.DataFrame([row])

    try:
        pred = float(model.predict(X)[0])
        st.success(f"Estimated {target_name.replace('_',' ').title()}: **{pred:,.2f}**")
        st.caption("Estimate from a scikit-learn Pipeline (One-Hot + Linear Regression).")
    except Exception as e:
        st.error("Prediction failed. Ensure model & schema match. Try redeploying with the latest files.")
        st.exception(e)

with st.expander("How it works"):
    st.markdown(
        """
        - **Model**: scikit-learn Pipeline with One-Hot Encoding for categoricals and a Linear Regression.
        - **Features**: Taken from your dataset (built dynamically from `schema.json`).
        - **Target**: Math score (or the closest available math column).
        - **Why small**: Linear Regression keeps the model artifact under the GitHub 25 MB limit (usually <1 MB).
        """
    )

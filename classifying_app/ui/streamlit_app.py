import streamlit as st
import requests

# Adjust these URLs if your containers use different ports or hostnames
BUILDER_API_URL = "http://localhost:8000"
CLASSIFIER_API_URL = "http://localhost:8001"

st.title("Naive Bayes Classifier UI")

# Health check for builder_app
st.header("Checking builder_app health...")
try:
    health_resp = requests.get(f"{BUILDER_API_URL}/health")
    if health_resp.status_code != 200:
        st.error(f"builder_app health check failed with status code {health_resp.status_code}.")
        st.stop()
    health_json = health_resp.json()
    if not health_json.get("status", "").lower() == "ok":
        st.error(f"builder_app health check did not return OK: {health_json}")
        st.stop()
except requests.exceptions.RequestException as e:
    st.error(f"Could not connect to builder_app health endpoint: {e}")
    st.stop()

# 1. Fetch available datasets from builder_app
st.header("1. Using the trained dataset")
try:
    datasets_resp = requests.get(f"{BUILDER_API_URL}/list_datasets")
    datasets = datasets_resp.json().get("datasets", [])
    if not datasets:
        st.error("No datasets found in /data directory.")
        st.stop()
    # Automatically select the first dataset (or you can set a default)
    dataset = datasets[0]
    st.write(f"Using dataset: {dataset}")
except requests.exceptions.RequestException as e:
    st.error(f"An error occurred while connecting to the builder server: {e}")
    st.stop()

# 2. Fetch columns/features for the selected dataset from builder_app
if dataset:
    try:
        columns_resp = requests.get(
            f"{BUILDER_API_URL}/get_dataset_columns", params={"dataset": dataset}
        )
        columns_data = columns_resp.json()
        columns = columns_data.get("columns", [])
        if not columns:
            st.error(f"No columns found in {dataset}.")
            st.stop()
        st.write(f"Columns in {dataset}: {columns}")

        # 3. Let user select features and input values
        st.header("2. Input feature values")
        feature_inputs = {}
        for col in columns[:-1]:  # Assume last column is the target
            val = st.text_input(f"Value for {col}")
            feature_inputs[col] = val

        # 4. Submit for prediction to classifying_app
        if st.button("Predict"):
            new_example = feature_inputs
            try:
                classify_resp = requests.post(
                    f"{CLASSIFIER_API_URL}/classify",
                    json={"dataset": dataset, "new_example": new_example}
                )
                result = classify_resp.json()
                if result.get("status") == "success":
                    st.success(f"Predicted class: {result.get('predicted_class')}")
                else:
                    st.error(f"Prediction failed: {result.get('message')}")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred while connecting to the classifier server: {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching dataset columns: {e}") 
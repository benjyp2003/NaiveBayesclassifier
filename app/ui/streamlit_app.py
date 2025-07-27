import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("Naive Bayes Classifier UI")

try:
    # 1. Fetch available datasets
    st.header("1. Choose a dataset")
    datasets_resp = requests.get(f"{API_URL}/list_datasets")
    datasets = datasets_resp.json().get("datasets", [])
    if not datasets:
        st.error("No datasets found in /data directory.")
        st.stop()

except requests.exceptions.RequestException as e:
    st.error(f"An error occurred while connecting to the server: {e}")
    st.stop()

dataset = st.selectbox("Select a dataset", datasets)

# 2. Fetch columns/features for the selected dataset
if dataset:
    try:
        columns_resp = requests.get(f"{API_URL}/get_dataset_columns", params={"dataset": dataset})
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

        # 4. Submit for prediction
        if st.button("Predict"):
            # Prepare the new_example dict
            new_example = feature_inputs
            classify_resp = requests.post(f"{API_URL}/classify", json={"new_example": new_example})
            result = classify_resp.json()
            if result.get("status") == "success":
                st.success(f"Predicted class: {result.get('predicted_class')}")
            else:
                st.error(f"Prediction failed: {result.get('message')}")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching dataset columns: {e}")


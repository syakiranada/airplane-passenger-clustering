import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st
import joblib

# Load the pre-trained model, scaler, and PCA components
scaler = joblib.load('scaler.pkl')  # Assuming the scaler was trained on these features
pca_group1 = joblib.load('pca_group1.pkl')  # Example PCA for 'Group1'
pca_group2 = joblib.load('pca_group2.pkl')  # Example PCA for 'Group2'
pca_group3 = joblib.load('pca_group3.pkl')  # Example PCA for 'Group3'
kmeans = joblib.load('kmeans_model.pkl')  # Loaded pre-trained KMeans model

# Model training columns (expected columns)
expected_columns = [
    'Age', 'Class', 'Departure Arrival time convenient', 'Gate location', 
    'Leg room service', 'Checkin service', 'Group1_PC1', 'Group1_PC2', 
    'Group2_PC1', 'Group2_PC2', 'Group3_PC1', 'Group3_PC2'
]

def preprocess_inference(data):
    # Convert to DataFrame if not already
    if isinstance(data, dict):
        data = pd.DataFrame([data])

    # Ensure required columns are present
    required_columns = ['Age', 'Class', 'Departure Arrival time convenient', 
                        'Gate location', 'Leg room service', 'Checkin service', 
                        'Cleanliness', 'Inflight entertainment', 'Seat comfort', 
                        'Food and drink', 'Inflight wifi service', 'Ease of Online booking', 
                        'Online boarding', 'Inflight service', 'Baggage handling', 
                        'On-board service']

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Missing columns in input data: {', '.join(missing_columns)}")
        return None

    # Handle scaling and PCA separately:
    # Convert 'Class' to numerical
    class_mapping = {"Business": 2, "Eco Plus": 1, "Eco": 0}
    data['Class'] = data['Class'].map(class_mapping)

    # Extract features to scale for PCA
    features_to_scale = [
        'Age', 'Class', 'Departure Arrival time convenient', 'Gate location',
        'Leg room service', 'Checkin service', 'Cleanliness', 'Inflight entertainment',
        'Seat comfort', 'Food and drink', 'Inflight wifi service', 'Ease of Online booking',
        'Online boarding', 'Inflight service', 'Baggage handling', 'On-board service'
    ]

    # Scale the features
    scaled_data = scaler.transform(data[features_to_scale])

    # Apply PCA to the scaled features for each group
    pca_group1_result = pca_group1.transform(scaled_data[:, 6:10])  # Columns for 'Group1'
    pca_group2_result = pca_group2.transform(scaled_data[:, 10:13])  # Columns for 'Group2'
    pca_group3_result = pca_group3.transform(scaled_data[:, 13:])  # Columns for 'Group3'

    # Combine scaled features and PCA results into the final feature set
    final_scaled_data = pd.DataFrame(scaled_data[:, :6], columns=features_to_scale[:6])

    # Combine PCA results into the final dataframe
    pca_cols = ['Group1_PC1', 'Group1_PC2', 'Group2_PC1', 'Group2_PC2', 'Group3_PC1', 'Group3_PC2']
    pca_features = pd.DataFrame(np.hstack([pca_group1_result, pca_group2_result, pca_group3_result]), columns=pca_cols)

    # Concatenate the scaled features with PCA features
    final_data = pd.concat([final_scaled_data, pca_features], axis=1)

    # Ensure the correct order of columns (must match what was used in training)
    final_data = final_data[expected_columns]

    # Perform clustering prediction using the trained KMeans model
    cluster_prediction = kmeans.predict(final_data)

    # Add the cluster prediction to the final data for reference
    final_data['Predicted Cluster'] = cluster_prediction

    # Display the final features along with the predicted cluster
    st.write("Final features for inference:", final_data.columns.tolist())
    st.write("Cluster prediction for this input data:", cluster_prediction[0])

    return final_data

# Example of how to use the preprocess_inference function
input_data = {
    'Age': [25],
    'Class': ['Business'],
    'Departure Arrival time convenient': [1],
    'Gate location': ['A1'],
    'Leg room service': [4],
    'Checkin service': [5],
    'Cleanliness': [4],
    'Inflight entertainment': [3],
    'Seat comfort': [5],
    'Food and drink': [3],
    'Inflight wifi service': [2],
    'Ease of Online booking': [4],
    'Online boarding': [3],
    'Inflight service': [5],
    'Baggage handling': [4],
    'On-board service': [5]
}

# Pass the input data into the function
processed_data = preprocess_inference(input_data)

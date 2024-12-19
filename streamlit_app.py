import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title('Airplane Passenger Satisfaction Clustering')
st.header("Kelompok 4 Kelas D")

with st.expander("Anggota"):
    st.markdown("""
    -  Rania (24060122120013)  
    -  Happy Desita W (24060122120023)  
    -  Syakira Nada N (24060122130049)  
    -  Asy'syifa Shabrina M (24060122130055)  
    """)

# Load model, scaler, and PCA from training
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load('scaler.pkl')
pca_group1 = joblib.load('pca_group1.pkl')
pca_group2 = joblib.load('pca_group2.pkl')
pca_group3 = joblib.load('pca_group3.pkl')

def preprocess_inference(data):
    # Ensure data is a DataFrame
    if isinstance(data, dict):
        data = pd.DataFrame([data])

    # Rename columns to match the training dataset
    data.rename(columns={
        "Departure/Arrival time convenient": "Departure Arrival time convenient",
        "Gate location": "Gate location",
        "Leg room service": "Leg room service",
        "Checkin service": "Checkin service",
        "Cleanliness": "Cleanliness",
        "Inflight entertainment": "Inflight entertainment",
        "Seat comfort": "Seat comfort",
        "Food and drink": "Food and drink",
        "Inflight wifi service": "Inflight wifi service",
        "Ease of Online booking": "Ease of Online booking",
        "Online boarding": "Online boarding",
        "Inflight service": "Inflight service",
        "Baggage handling": "Baggage handling",
        "On-board service": "On-board service"
    }, inplace=True)

    # Map Class values exactly as in training
    class_mapping = {"Business": 2, "Eco Plus": 1, "Eco": 0}
    data['Class'] = data['Class'].map(class_mapping)

    # Define feature groups for PCA and scaling
    group1 = ['Cleanliness', 'Inflight entertainment', 'Seat comfort', 'Food and drink']
    group2 = ['Inflight wifi service', 'Ease of Online booking', 'Online boarding']
    group3 = ['Inflight service', 'Baggage handling', 'On-board service']

    # Numerical features for scaling
    numerical_features = [
        'Age',
        'Class',
        'Departure Arrival time convenient',
        'Gate location',
        'Leg room service',
        'Checkin service'
    ]

    # Scale the features before applying PCA
    scaled_data_group1 = scaler.transform(data[group1])
    scaled_data_group2 = scaler.transform(data[group2])
    scaled_data_group3 = scaler.transform(data[group3])

    # Apply PCA transformation
    group1_pca = pca_group1.transform(scaled_data_group1)
    group2_pca = pca_group2.transform(scaled_data_group2)
    group3_pca = pca_group3.transform(scaled_data_group3)

    # Combine original features with PCA results
    pca_cols = ['Group1_PC1', 'Group1_PC2', 'Group2_PC1', 'Group2_PC2', 'Group3_PC1', 'Group3_PC2']
    pca_features = pd.DataFrame(np.hstack([group1_pca, group2_pca, group3_pca]), columns=pca_cols)

    final_features = pd.DataFrame(data[numerical_features].values, columns=numerical_features)
    
    # Concatenate the scaled features and PCA results
    final_features = pd.concat([final_features, pca_features], axis=1)

    return final_features

st.header("Masukkan Data untuk Klasifikasi")

with st.form("data_input_form"):
    Class = st.selectbox("Class", ["Business", "Eco Plus", "Eco"])
    Age = st.number_input("Age", min_value=0, max_value=100, value=25)
    Departure_Arrival_time_convenient = st.slider("Departure/Arrival time convenient", 0, 5)
    Gate_location = st.slider("Gate location", 1, 5)
    Leg_room_service = st.slider("Leg room service", 0, 5)
    Checkin_service = st.slider("Checkin service", 1, 5)
    
    Cleanliness = st.slider("Cleanliness", 0, 5)
    Inflight_entertainment = st.slider("Inflight entertainment", 0, 5)
    Seat_comfort = st.slider("Seat comfort", 1, 5)
    Food_and_drink = st.slider("Food and drink", 0, 5)
    
    Inflight_wifi_service = st.slider("Inflight wifi service", 1, 5)
    Ease_of_Online_booking = st.slider("Ease of Online booking", 0, 5)
    Online_boarding = st.slider("Online boarding", 0, 5)
    
    Inflight_service = st.slider("Inflight service", 0, 5)
    Baggage_handling = st.slider("Baggage handling", 1, 5)
    On_board_service = st.slider("On-board service", 0, 5)
    
    submitted = st.form_submit_button("Submit")

if submitted:
    user_data = {
        "Class": Class,
        "Age": Age,
        "Departure/Arrival time convenient": Departure_Arrival_time_convenient,
        "Gate location": Gate_location,
        "Leg room service": Leg_room_service,
        "Checkin service": Checkin_service,
        "Cleanliness": Cleanliness,
        "Inflight entertainment": Inflight_entertainment,
        "Seat comfort": Seat_comfort,
        "Food and drink": Food_and_drink,
        "Inflight wifi service": Inflight_wifi_service,
        "Ease of Online booking": Ease_of_Online_booking,
        "Online boarding": Online_boarding,
        "Inflight service": Inflight_service,
        "Baggage handling": Baggage_handling,
        "On-board service": On_board_service
    }
    
    try:
        # Preprocess the input data
        processed_data = preprocess_inference(user_data)
        
        # Predict the cluster
        cluster = kmeans.predict(processed_data)[0]
        st.success(f"Data Anda termasuk dalam Cluster: {cluster}")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load pre-trained models and scalers
# scaler = joblib.load('scaler.pkl')
pca_group1 = joblib.load('pca_group1.pkl')
pca_group2 = joblib.load('pca_group2.pkl')
pca_group3 = joblib.load('pca_group3.pkl')
kmeans = joblib.load('kmeans_model.pkl')

from sklearn.preprocessing import MinMaxScaler

# Title and Description
st.title("Passenger Clustering App")
st.write("Aplikasi ini akan memprediksi cluster untuk data penumpang berdasarkan input fitur Anda.")

# Input Form
with st.form("clustering_form"):
    age = st.number_input("Masukkan Usia (Age)", min_value=0, max_value=100, step=1)
    flight_class = st.selectbox("Pilih Kelas Penerbangan (Class)", ["Business", "Eco Plus", "Eco"])
    departure_convenience = st.slider("Kemudahan Waktu Keberangkatan dan Kedatangan", 0, 5, 3)
    gate_location = st.slider("Lokasi Gerbang (Gate Location)", 0, 5, 3)
    leg_room_service = st.slider("Layanan Ruang Kaki (Leg Room Service)", 0, 5, 3)
    checkin_service = st.slider("Layanan Check-in", 0, 5, 3)
    cleanliness = st.slider("Kebersihan (Cleanliness)", 0, 5, 3)
    inflight_entertainment = st.slider("Hiburan di Pesawat (Inflight Entertainment)", 0, 5, 3)
    seat_comfort = st.slider("Kenyamanan Kursi (Seat Comfort)", 0, 5, 3)
    food_and_drink = st.slider("Makanan dan Minuman (Food and Drink)", 0, 5, 3)
    inflight_wifi = st.slider("Wi-Fi di Pesawat (Inflight WiFi)", 0, 5, 3)
    online_booking = st.slider("Kemudahan Pemesanan Online (Ease of Online Booking)", 0, 5, 3)
    online_boarding = st.slider("Proses Boarding Online", 0, 5, 3)
    inflight_service = st.slider("Layanan di Pesawat (Inflight Service)", 0, 5, 3)
    baggage_handling = st.slider("Penanganan Bagasi (Baggage Handling)", 0, 5, 3)
    onboard_service = st.slider("Layanan di Pesawat (On-board Service)", 0, 5, 3)
    
    submitted = st.form_submit_button("Submit")

if submitted:
    # Map flight class to numerical values
    class_mapping = {"Business": 2, "Eco Plus": 1, "Eco": 0}
    flight_class = class_mapping[flight_class]

    # Prepare input data
    input_data = {
        'Age': age,
        'Class': flight_class,
        'Departure Arrival time convenient': departure_convenience,
        'Gate location': gate_location,
        'Leg room service': leg_room_service,
        'Checkin service': checkin_service,
        'Cleanliness': cleanliness,
        'Inflight entertainment': inflight_entertainment,
        'Seat comfort': seat_comfort,
        'Food and drink': food_and_drink,
        'Inflight wifi service': inflight_wifi,
        'Ease of Online booking': online_booking,
        'Online boarding': online_boarding,
        'Inflight service': inflight_service,
        'Baggage handling': baggage_handling,
        'On-board service': onboard_service
    }

    data_df = pd.DataFrame([input_data])

    # Scale and process the data (excluding specific features from scaling)
    features_to_scale = [
        'Cleanliness', 'Inflight entertainment', 'Seat comfort', 'Food and drink',
        'Inflight wifi service', 'Ease of Online booking', 'Online boarding',
        'Inflight service', 'Baggage handling', 'On-board service'
    ]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.transform(data_df[features_to_scale])
    
    # Combine unscaled and scaled features
    unscaled_features = data_df[['Age', 'Class', 'Departure Arrival time convenient', 'Gate location', 'Leg room service', 'Checkin service']].values
    combined_features = np.hstack([unscaled_features, scaled_features])
    
    # Validate dimensions
    st.write(f"Shape of combined features: {combined_features.shape}")
    
    # Define indices for PCA groups based on combined_features
    # Adjust these indices based on your actual PCA training
    pca_group1_features = combined_features[:, 6:10]  # Group1: Features 6-9
    pca_group2_features = combined_features[:, 10:13]  # Group2: Features 10-12
    pca_group3_features = combined_features[:, 13:]  # Group3: Features 13+
    
    # Apply PCA for each group
    pca_group1_result = pca_group1.transform(pca_group1_features)
    pca_group2_result = pca_group2.transform(pca_group2_features)
    pca_group3_result = pca_group3.transform(pca_group3_features)
    
    # Combine results
    pca_features = np.hstack([pca_group1_result, pca_group2_result, pca_group3_result])
    
    # Include unscaled features and PCA features for final prediction
    final_data = np.hstack([unscaled_features, pca_features])
    
    # Validate dimensions of final_data
    st.write(f"Shape of final data for clustering: {final_data.shape}")
    
    # Predict cluster
    cluster = kmeans.predict(final_data)[0]
    
    # Display the result
    st.write(f"Data Anda masuk ke dalam Cluster: {cluster}")

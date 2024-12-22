import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load pre-trained models and scalers
# scaler = joblib.load('scaler.pkl')
pca_group1 = joblib.load('pca_group1.pkl')
pca_group2 = joblib.load('pca_group2.pkl')
pca_group3 = joblib.load('pca_group3.pkl')
kmeans = joblib.load('kmeans_model.pkl')

scaler_0_1 = MinMaxScaler()  # Assuming scaler for 0-1 scaling
scaler_0_5 = MinMaxScaler(feature_range=(0, 5))  # Assuming scaler for 0-5 scaling

# Title and Description
st.title("Passenger Clustering App")
st.write("Aplikasi ini akan memprediksi cluster untuk data penumpang berdasarkan input fitur Anda.")

with st.expander("Kelompok 4 Kelas D"):
      # Anggota
      st.markdown("""
      -  Rania (24060122120013)  
      -  Happy Desita W (24060122120023)  
      -  Syakira Nada N (24060122130049)  
      -  Asy’syifa Shabrina M (24060122130055)  
        """)

# Input Form
with st.form("clustering_form"):
    age = st.number_input("Masukkan Usia (Age)", min_value=0, max_value=100, step=1, value=25)
    flight_class = st.selectbox("Pilih Kelas Penerbangan (Class)", ["Business", "Eco Plus", "Eco"], index=2)
    departure_convenience = st.slider("Kemudahan Waktu Keberangkatan dan Kedatangan (Departure/Arrival time convenient)", 0, 5, 3)
    gate_location = st.slider("Lokasi Gerbang (Gate Location)", 0, 5, 3)
    leg_room_service = st.slider("Layanan Ruang Kaki (Leg Room Service)", 0, 5, 3)
    cleanliness = st.slider("Kebersihan (Cleanliness)", 0, 5, 3)
    inflight_entertainment = st.slider("Hiburan di Pesawat (Inflight Entertainment)", 0, 5, 3)
    seat_comfort = st.slider("Kenyamanan Kursi (Seat Comfort)", 0, 5, 3)
    food_and_drink = st.slider("Makanan dan Minuman (Food and Drink)", 0, 5, 3)
    inflight_wifi = st.slider("Wi-Fi di Pesawat (Inflight WiFi service)", 0, 5, 3)
    online_booking = st.slider("Kemudahan Pemesanan Online (Ease of Online Booking)", 0, 5, 3)
    online_boarding = st.slider("Proses Boarding Online (Online boarding)", 0, 5, 3)
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
        'Departure/Arrival time convenient': departure_convenience,
        'Gate location': gate_location,
        'Leg room service': leg_room_service,
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

    df_input = pd.DataFrame([input_data])

    # 1. Scale features to 0-1 for specific columns
    features_to_scale_0_1 = [
        'Cleanliness', 'Inflight entertainment', 'Seat comfort', 'Food and drink',
        'Inflight wifi service', 'Ease of Online booking', 'Online boarding',
        'Inflight service', 'Baggage handling', 'On-board service'
    ]
    scaled_features_0_1 = scaler_0_1.fit_transform(df_input[features_to_scale_0_1])

    # Combine unscaled features (non-scaled columns)
    unscaled_features = df_input[['Departure/Arrival time convenient', 'Gate location', 'Leg room service']].values
    
    # Normalize Age and Class columns to range 0-5
    scaled_age_class = scaler_0_5.fit_transform(df_input[['Age', 'Class']])

    # Combine scaled Age/Class features with unscaled ones
    combined_features = np.hstack([scaled_age_class, unscaled_features, scaled_features_0_1])

    # 2. Apply PCA for each group (Group 1, 2, 3)
    pca_group1_result = pca_group1.transform(combined_features[:, 5:9])  # Group 1 features
    pca_group2_result = pca_group2.transform(combined_features[:, 9:12])  # Group 2 features
    pca_group3_result = pca_group3.transform(combined_features[:, 12:])  # Group 3 features

    # Combine PCA results
    pca_features = np.hstack([pca_group1_result, pca_group2_result, pca_group3_result])

    # 3. Normalize PCA results to range 0-5
    scaled_pca_features = scaler_0_5.fit_transform(pca_features)

    # 4. Combine unscaled features and scaled PCA features
    final_data = np.hstack([combined_features[:, :5], scaled_pca_features])

    # Predict cluster using KMeans
    cluster = kmeans.predict(final_data)[0]

    # Display the result
    st.write(f"## Data Anda masuk ke dalam Cluster: {cluster}")

    # Profiling for each cluster
    if cluster == 0:
        st.write("### Profil Cluster 0:")
        st.write("Cluster 0 didominasi oleh penumpang dengan rata-rata usia lebih tua, lebih memilih layanan premium (eco plus atau business) dengan tingkat kenyamanan tinggi dalam aspek kursi, kebersihan, dan layanan ruang kaki. Mereka juga menghargai kemudahan dalam pemesanan tiket dan efisiensi layanan bagasi. Fokus utama adalah kenyamanan fisik selama penerbangan dan kualitas layanan.")
    elif cluster == 1:
        st.write("### Profil Cluster 1:")
        st.write("Cluster 1 terdiri dari penumpang yang lebih muda, lebih memilih kelas ekonomi dengan preferensi pada layanan yang hemat biaya. Meskipun mereka memberi perhatian terhadap kenyamanan kursi dan kebersihan, mereka lebih menghargai kemudahan dalam pengalaman digital seperti pemesanan online dan proses boarding. Layanan bagasi juga menjadi perhatian penting bagi mereka.")

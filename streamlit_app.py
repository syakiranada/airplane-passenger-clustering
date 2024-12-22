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
        'Departure Arrival time convenient': departure_convenience,
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

    data_df = pd.DataFrame([input_data])

    # 1. Skalakan fitur sebelum PCA dengan MinMaxScaler default (0, 1)
    scaler_before_pca = MinMaxScaler()
    features_before_pca = [
        'Cleanliness', 'Inflight entertainment', 'Seat comfort', 'Food and drink',
        'Inflight wifi service', 'Ease of Online booking', 'Online boarding',
        'Inflight service', 'Baggage handling', 'On-board service'
    ]
    scaled_features = scaler_before_pca.fit_transform(df_input[features_before_pca])

    # 2. Terapkan PCA pada fitur yang sudah dinormalisasi
    group1_features = scaled_features[:, :4]  # Fitur untuk Group1 (kolom 1-4)
    group2_features = scaled_features[:, 4:7]  # Fitur untuk Group2 (kolom 5-7)
    group3_features = scaled_features[:, 7:]  # Fitur untuk Group3 (kolom 8-10)

    group1_pca = pca_group1.transform(group1_features)
    group2_pca = pca_group2.transform(group2_features)
    group3_pca = pca_group3.transform(group3_features)

    # Gabungkan hasil PCA
    pca_results = np.hstack([group1_pca, group2_pca, group3_pca])

    # 3. Gabungkan kolom tambahan dengan hasil PCA
    columns_to_keep = ['Age', 'Class', 'Departure/Arrival time convenient', 'Gate location', 'Leg room service']
    data_for_scaling = df_input[['Age', 'Class']].values  # Hanya kolom Age dan Class yang akan diskalakan
    final_data = np.hstack([data_for_scaling, df_input[columns_to_keep[2:]].values, pca_results])

    # 4. Skalakan hanya kolom Age, Class, dan hasil PCA dengan MinMaxScaler (0, 5)
    scaler_final = MinMaxScaler(feature_range=(0, 5))
    scaled_part = scaler_final.fit_transform(np.hstack([data_for_scaling, pca_results]))
    final_scaled_data = np.hstack([scaled_part, df_input[columns_to_keep[2:]].values])  # Gabungkan dengan kolom tambahan

    # Buat DataFrame final
    final_columns = [
        'Age', 'Class', 'Departure/Arrival time convenient', 'Gate location', 'Leg room service',
        'Group1_PC1', 'Group1_PC2', 'Group2_PC1', 'Group2_PC2', 'Group3_PC1', 'Group3_PC2'
    ]
    df_final = pd.DataFrame(final_scaled_data, columns=final_columns)
      
    # Tampilkan hasil
    st.write(df_final)
      
    # Predict cluster
    cluster = kmeans.predict(final_scaled_data)[0]

    # Display the result
    st.write(f"## Data Anda masuk ke dalam Cluster: {cluster}")

    # Profiling for each cluster
    if cluster == 0:
        st.write("### Profil Cluster 0:")
        st.write("Cluster 0: Penumpang lebih tua (usia rata-rata 51.81 tahun), cenderung memilih kelas eco plus, dan fokus pada kenyamanan kursi, kebersihan, dan layanan ruang kaki. Pentingnya kemudahan pemesanan online dan efisiensi layanan bagasi juga menonjol.")
    elif cluster == 1:
        st.write("### Profil Cluster 1:")
        st.write("Cluster 1: Penumpang muda (usia rata-rata 26.61 tahun), lebih sering memilih kelas ekonomi, dan menghargai kenyamanan kursi, layanan digital seperti pemesanan online, dan keandalan layanan bagasi.")

import streamlit as st
import pandas as pd
import numpy as np  # Untuk operasi numerik
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

st.title('Airplane Passenger Satisfaction Clustering')
# This is The Header
st.header("Kelompok 4 Kelas D")

with st.expander("**Anggota**"):
      # Anggota
      st.markdown("""
      -  Rania (24060122120013)  
      -  Happy Desita W (24060122120023)  
      -  Syakira Nada N (24060122130049)  
      -  Asy’syifa Shabrina M (24060122130055)  
        """)

# Load model, scaler dan PCA dari training
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load('scaler.pkl')  # Scaler yang disimpan saat training
pca_group1 = joblib.load('pca_group1.pkl')  # PCA Group1 dari training
pca_group2 = joblib.load('pca_group2.pkl')  # PCA Group2 dari training
pca_group3 = joblib.load('pca_group3.pkl')  # PCA Group3 dari training

def preprocess_inference(data):
    # Ensure data is a DataFrame
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    # Rename columns if scaler was trained with underscores
    data.rename(columns={
        "Inflight entertainment": "Inflight_entertainment",
        "Seat comfort": "Seat_comfort",
        "Food and drink": "Food_and_drink",
        "Inflight wifi service": "Inflight_wifi_service",
        "Ease of Online booking": "Ease_of_Online_booking",
        "On-board service": "On_board_service"
    }, inplace=True)
    
    # Mapping for 'Class'
    class_mapping = {"Business": 2, "Eco Plus": 1, "Eco": 0}
    if 'Class' in data.columns:
        data['Class'] = data['Class'].map(class_mapping)
    
    # Define feature groups
    group1 = ['Cleanliness', 'Inflight_entertainment', 'Seat_comfort', 'Food_and_drink']
    group2 = ['Inflight_wifi_service', 'Ease_of_Online_booking', 'Online_boarding']
    group3 = ['Inflight_service', 'Baggage_handling', 'On_board_service']
    
    # Validate input columns
    required_columns = group1 + group2 + group3
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in input data: {missing_columns}")
    
    # Scale each group with respective scaler
    data_group1 = scaler_group1.transform(data[group1])
    data_group2 = scaler_group2.transform(data[group2])
    data_group3 = scaler_group3.transform(data[group3])
    
    # Apply PCA
    group1_pca = pca_group1.transform(data_group1)
    group2_pca = pca_group2.transform(data_group2)
    group3_pca = pca_group3.transform(data_group3)
    
    # Convert PCA results to DataFrames
    df_group1_pca = pd.DataFrame(group1_pca, columns=['Group1_PC1', 'Group1_PC2'])
    df_group2_pca = pd.DataFrame(group2_pca, columns=['Group2_PC1', 'Group2_PC2'])
    df_group3_pca = pd.DataFrame(group3_pca, columns=['Group3_PC1', 'Group3_PC2'])
    
    # Combine PCA results with original data
    data.reset_index(drop=True, inplace=True)
    df_pca_combined = pd.concat([data, df_group1_pca, df_group2_pca, df_group3_pca], axis=1)
    
    # Drop original features
    df_pca_combined = df_pca_combined.drop(columns=required_columns)
    
    return df_pca_combined


st.header("Masukkan Data untuk Klasifikasi")
# Form input pengguna
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
    
    # Submit tombol
    submitted = st.form_submit_button("Submit")

# Jika form disubmit
if submitted:
    # Buat dictionary dari input pengguna
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
    
    # Preprocessing
    processed_data = preprocess_inference(user_data)
    
    # Debug: Menampilkan data yang telah diproses
    st.write(processed_data)
    
    # Prediksi cluster
    cluster = kmeans.predict(processed_data)[0]
    st.write(f"Data Anda termasuk dalam Cluster: {cluster}")

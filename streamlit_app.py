import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

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

st.write("Expected features from model:", kmeans.feature_names_in_.tolist())

def preprocess_inference(data):
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    # Exact column order from CSV
    expected_columns = [
        'Class',
        'Age',
        'Departure/Arrival time convenient',
        'Gate location',
        'Leg room service',
        'Checkin service',
        'Cleanliness',
        'Inflight entertainment',
        'Seat comfort',
        'Food and drink',
        'Inflight wifi service',
        'Ease of Online booking',
        'Online boarding',
        'Inflight service',
        'Baggage handling',
        'On-board service'
    ]
    
    # Ensure columns are in correct order
    data = data[expected_columns]
    
    # Convert Class to numeric
    class_mapping = {"Business": 2, "Eco Plus": 1, "Eco": 0}
    data['Class'] = data['Class'].map(class_mapping)
    
    # Define groups for PCA
    group1 = ['Cleanliness', 'Inflight entertainment', 'Seat comfort', 'Food and drink']
    group2 = ['Inflight wifi service', 'Ease of Online booking', 'Online boarding']
    group3 = ['Inflight service', 'Baggage handling', 'On-board service']
    
    # Scale and transform groups
    group1_data = scaler.transform(data[group1])
    group2_data = scaler.transform(data[group2])
    group3_data = scaler.transform(data[group3])
    
    # Apply PCA
    group1_pca = pca_group1.transform(group1_data)
    group2_pca = pca_group2.transform(group2_data)
    group3_pca = pca_group3.transform(group3_data)
    
    # Create DataFrame with numerical features
    numerical_features = [
        'Age',
        'Class',
        'Departure/Arrival time convenient',
        'Gate location',
        'Leg room service',
        'Checkin service'
    ]
    result_df = data[numerical_features].copy()
    
    # Add PCA components
    for i, pca_data in enumerate(group1_pca[0]):
        result_df[f'Group1_PC{i+1}'] = pca_data
    for i, pca_data in enumerate(group2_pca[0]):
        result_df[f'Group2_PC{i+1}'] = pca_data
    for i, pca_data in enumerate(group3_pca[0]):
        result_df[f'Group3_PC{i+1}'] = pca_data
    
    return result_df

st.header("Masukkan Data untuk Klasifikasi")

with st.form("data_input_form"):
    # Form inputs sesuai urutan CSV
    Class = st.selectbox("Class", ["Business", "Eco Plus", "Eco"])
    Age = st.number_input("Age", min_value=0, max_value=100, value=25)
    Departure_Arrival_time_convenient = st.slider("Departure/Arrival time convenient", 0, 5, 0)
    Gate_location = st.slider("Gate location", 0, 5, 1)
    Leg_room_service = st.slider("Leg room service", 0, 5, 0)
    Checkin_service = st.slider("Checkin service", 0, 5, 1)
    
    Cleanliness = st.slider("Cleanliness", 0, 5, 0)
    Inflight_entertainment = st.slider("Inflight entertainment", 0, 5, 0)
    Seat_comfort = st.slider("Seat comfort", 0, 5, 1)
    Food_and_drink = st.slider("Food and drink", 0, 5, 0)
    
    Inflight_wifi_service = st.slider("Inflight wifi service", 0, 5, 1)
    Ease_of_Online_booking = st.slider("Ease of Online booking", 0, 5, 0)
    Online_boarding = st.slider("Online boarding", 0, 5, 0)
    
    Inflight_service = st.slider("Inflight service", 0, 5, 0)
    Baggage_handling = st.slider("Baggage handling", 0, 5, 1)
    On_board_service = st.slider("On-board service", 0, 5, 0)
    
    submitted = st.form_submit_button("Submit")

if submitted:
    # Create input data dalam urutan yang sama dengan CSV
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
        # Debug: tampilkan data input
        st.write("Input features:")
        input_df = pd.DataFrame([user_data])
        st.write(input_df)
        
        # Proses data
        processed_data = preprocess_inference(user_data)
        
        # Debug: tampilkan data yang telah diproses
        st.write("Processed features:")
        st.write(processed_data)
        
        # Debug: tampilkan nama-nama fitur
        st.write("Feature names in processed data:", processed_data.columns.tolist())
        
        # Prediksi
        cluster = kmeans.predict(processed_data)[0]
        st.success(f"Data Anda termasuk dalam Cluster: {cluster}")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Debug information:")
        st.write(f"Available columns: {list(processed_data.columns) if 'processed_data' in locals() else 'No processed data available'}")

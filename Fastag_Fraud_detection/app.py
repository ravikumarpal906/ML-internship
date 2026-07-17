import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# 1. Page Configuration
st.set_page_config(page_title="Fastag Fraud Detection", page_icon="💳", layout="centered")

# 2. Load the trained Gradient Boosting model
@st.cache_resource
def load_model():
    return joblib.load("fastag_fraud_model.pkl")

# 3. Label Encoding Mappings (derived directly from your dataset)
vehicle_type_map = {'Bus': 0, 'Car': 1, 'Motorcycle': 2, 'SUV': 3, 'Sedan': 4, 'Truck': 5, 'Van': 6}
tollbooth_map = {'A-101': 0, 'B-102': 1, 'C-103': 2, 'D-104': 3, 'D-105': 4, 'D-106': 5}
lane_type_map = {'Express': 0, 'Regular': 1}
vehicle_dim_map = {'Large': 0, 'Medium': 1, 'Small': 2}
state_code_map = {'AP': 0, 'BR': 1, 'DL': 2, 'GA': 3, 'GJ': 4, 'HR': 5, 'KA': 6, 'KL': 7, 
                  'MH': 8, 'MP': 9, 'RJ': 10, 'TN': 11, 'TS': 12, 'UP': 13, 'WB': 14}

def main():
    st.title("💳 Fastag Fraud Detection Dashboard")
    st.write("Enter the transaction details below to verify if it is legitimate using our Gradient Boosting Model.")

    try:
        model = load_model()
    except FileNotFoundError:
        st.error("Model file 'fastag_fraud_model.pkl' not found. Ensure it is in the same folder as this script.")
        st.stop()

    st.sidebar.header("Transaction Parameters")

    # 4. User Inputs using Selectboxes (matching your exact dataset options)
    vehicle_type = st.sidebar.selectbox("Vehicle Type", list(vehicle_type_map.keys()))
    toll_booth = st.sidebar.selectbox("Toll Booth ID", list(tollbooth_map.keys()))
    lane_type = st.sidebar.selectbox("Lane Type", list(lane_type_map.keys()))
    vehicle_dimensions = st.sidebar.selectbox("Vehicle Dimensions", list(vehicle_dim_map.keys()))
    
    transaction_amount = st.sidebar.number_input("Transaction Amount (Billed)", min_value=0, value=350, step=10)
    amount_paid = st.sidebar.number_input("Amount Paid", min_value=0, value=120, step=10)
    vehicle_speed = st.sidebar.number_input("Vehicle Speed (km/h)", min_value=0, value=65)
    
    state_code = st.sidebar.selectbox("State Code", list(state_code_map.keys()), index=6) # Default to KA
    
    date_input = st.sidebar.date_input("Transaction Date", datetime.date(2023, 1, 6))
    time_input = st.sidebar.time_input("Transaction Time", datetime.time(11, 20))

    # 5. Extract Time Features
    hour = time_input.hour
    day_of_week = date_input.weekday() # Monday=0, Sunday=6
    month = date_input.month

    # Display Human-Readable Summary
    st.subheader("Current Input Summary")
    df_display = pd.DataFrame({
        "Vehicle Type": [vehicle_type],
        "Toll Booth": [toll_booth],
        "Lane": [lane_type],
        "Dimensions": [vehicle_dimensions],
        "Billed": [transaction_amount],
        "Paid": [amount_paid],
        "Speed": [vehicle_speed],
        "State Code": [state_code],
        "Hour": [hour],
        "Day": [day_of_week],
        "Month": [month]
    })
    st.dataframe(df_display, hide_index=True)

    # 6. Apply Encodings and Predict
    if st.button("Analyze Transaction Validity", type="primary"):
        
        # Build the X dataframe in the exact order your GradientBoosting model expects
        X_input = pd.DataFrame({
            'Vehicle_Type': [vehicle_type_map[vehicle_type]],
            'TollBoothID': [tollbooth_map[toll_booth]],
            'Lane_Type': [lane_type_map[lane_type]],
            'Vehicle_Dimensions': [vehicle_dim_map[vehicle_dimensions]],
            'Transaction_Amount': [transaction_amount],
            'Amount_paid': [amount_paid],
            'Vehicle_Speed': [vehicle_speed],
            'state_code': [state_code_map[state_code]],
            'Hour': [hour],
            'Day_of_Week': [day_of_week],
            'Month': [month]
        })

        try:
            # Predict
            prediction = model.predict(X_input)
            
            # Display Results
            st.subheader("Analysis Result")
            
            # Model output logic: 0 = Fraud, 1 = Not Fraud
            if prediction[0] == 0:
                st.error("🚨 **Fraudulent Transaction Detected!**")
                st.warning("The Gradient Boosting model classified this transaction as an anomaly.")
            else:
                st.success("✅ **Transaction Appears Valid (Not Fraud)**")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
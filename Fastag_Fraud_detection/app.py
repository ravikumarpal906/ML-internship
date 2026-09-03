import streamlit as st
import pandas as pd
import joblib
import datetime

# ---------------------------------------------------------------------------
# Fixes vs. the original app.py:
#   - No more hand-typed category -> integer dictionaries. Those are loaded
#     from label_encoders.pkl (produced by train_model.py), so they can
#     never silently drift out of sync with what the model was trained on.
#   - Dropdown options are read from the encoders themselves, so the UI can
#     never offer a category the model doesn't know about.
#   - The 0="Fraud"/1="Not Fraud" assumption is replaced with the actual
#     target_encoder, so the label shown is always correct even if class
#     codes ever change.
#   - Includes only the features train_model.py actually uses (no leaky
#     Payment_Diff/Payment_Ratio derived columns).
#   - Builds the inference row using the saved feature_columns order, so
#     column-order mismatches can't happen even if the pipeline changes later.
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Fastag Fraud Detection", page_icon="💳", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("fastag_fraud_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, label_encoders, target_encoder, feature_columns


def encode_category(value, col, label_encoders):
    mapping = label_encoders[col]
    return mapping["class_to_code"].get(value, mapping["unknown_code"])


def main():
    st.title("💳 Fastag Fraud Detection Dashboard")
    st.write("Enter the transaction details below to check it with the trained model.")

    try:
        model, label_encoders, target_encoder, feature_columns = load_artifacts()
    except FileNotFoundError as e:
        st.error(
            f"Missing artifact file: {e}. Make sure fastag_fraud_model.pkl, "
            "label_encoders.pkl, target_encoder.pkl, and feature_columns.pkl "
            "are all in this folder (run train_model.py first to produce them)."
        )
        st.stop()

    st.sidebar.header("Transaction Parameters")

    # Options come straight from the training data's encoders.
    vehicle_type_options = list(label_encoders["Vehicle_Type"]["class_to_code"].keys())
    tollbooth_options = list(label_encoders["TollBoothID"]["class_to_code"].keys())
    lane_type_options = list(label_encoders["Lane_Type"]["class_to_code"].keys())
    vehicle_dim_options = list(label_encoders["Vehicle_Dimensions"]["class_to_code"].keys())
    state_code_options = list(label_encoders["state_code"]["class_to_code"].keys())

    vehicle_type = st.sidebar.selectbox("Vehicle Type", vehicle_type_options)
    toll_booth = st.sidebar.selectbox("Toll Booth ID", tollbooth_options)
    lane_type = st.sidebar.selectbox("Lane Type", lane_type_options)
    vehicle_dimensions = st.sidebar.selectbox("Vehicle Dimensions", vehicle_dim_options)

    transaction_amount = st.sidebar.number_input("Transaction Amount (Billed)", min_value=0, value=350, step=10)
    amount_paid = st.sidebar.number_input("Amount Paid", min_value=0, value=120, step=10)
    vehicle_speed = st.sidebar.number_input("Vehicle Speed (km/h)", min_value=0, value=65)

    state_code = st.sidebar.selectbox(
        "State Code", state_code_options,
        index=state_code_options.index("KA") if "KA" in state_code_options else 0,
    )

    date_input = st.sidebar.date_input("Transaction Date", datetime.date(2023, 1, 6))
    time_input = st.sidebar.time_input("Transaction Time", datetime.time(11, 20))

    hour = time_input.hour
    day_of_week = date_input.weekday()
    month = date_input.month

    st.subheader("Current Input Summary")
    st.dataframe(
        pd.DataFrame({
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
            "Month": [month],
        }),
        hide_index=True,
    )

    if st.button("Analyze Transaction Validity", type="primary"):
        row = {
            "Vehicle_Type": encode_category(vehicle_type, "Vehicle_Type", label_encoders),
            "TollBoothID": encode_category(toll_booth, "TollBoothID", label_encoders),
            "Lane_Type": encode_category(lane_type, "Lane_Type", label_encoders),
            "Vehicle_Dimensions": encode_category(vehicle_dimensions, "Vehicle_Dimensions", label_encoders),
            "Transaction_Amount": transaction_amount,
            "Amount_paid": amount_paid,
            "Vehicle_Speed": vehicle_speed,
            "state_code": encode_category(state_code, "state_code", label_encoders),
            "Hour": hour,
            "Day_of_Week": day_of_week,
            "Month": month,
        }

        # FIX: build the row in the EXACT column order used at training time,
        # loaded from feature_columns.pkl instead of assumed/re-typed here.
        X_input = pd.DataFrame([row])[feature_columns]

        try:
            prediction = model.predict(X_input)[0]
            predicted_label = target_encoder.inverse_transform([prediction])[0]

            st.subheader("Analysis Result")
            if predicted_label == "Fraud":
                st.error("🚨 **Fraudulent Transaction Detected!**")
            else:
                st.success("✅ **Transaction Appears Valid (Not Fraud)**")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_input)[0]
                fraud_idx = list(target_encoder.classes_).index("Fraud")
                st.caption(f"Model confidence — Fraud: {proba[fraud_idx]:.1%}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
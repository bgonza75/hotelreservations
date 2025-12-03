
import streamlit as st
import pickle
import pandas as pd

# ------------------------------
# Load Models and Metadata
# ------------------------------
with open("logistic_regression_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("decision_tree_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

with open("data_info.pkl", "rb") as f:
    data_info = pickle.load(f)

# Load scaler (NEW REQUIRED FIX)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

expected_columns = data_info["expected_columns"]
numeric_columns = data_info["raw_numeric_columns"]
categorical_columns = data_info["raw_categorical_columns"]

st.title("🏨 Hotel Reservation Cancellation Predictor")
st.write("Predict whether a hotel booking will be **Canceled** or **Not Canceled**.")

# ------------------------------
# USER INPUT FORM
# ------------------------------

st.header("Enter Reservation Details")

user_inputs = {}

# Numeric Inputs (sliders)
for col in numeric_columns:
    user_inputs[col] = st.number_input(col.replace("_", " ").title(),
                                      min_value=0.0, value=1.0, step=1.0)

# Categorical Inputs
meal_options = ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]
room_options = ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]
market_segment_options = ["Offline", "Online", "Corporate", "Complementary", "Aviation"]

user_inputs["type_of_meal_plan"] = st.selectbox("Meal Plan", meal_options)
user_inputs["room_type_reserved"] = st.selectbox("Room Type Reserved", room_options)
user_inputs["market_segment_type"] = st.selectbox("Market Segment", market_segment_options)

# Convert to DataFrame
input_df = pd.DataFrame([user_inputs])

# ------------------------------
# One-Hot Encode to Match Training
# ------------------------------
input_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)

# Add missing columns
for col in expected_columns:
    if col not in input_encoded:
        input_encoded[col] = 0

# Reorder columns
input_encoded = input_encoded[expected_columns]

# ------------------------------
# Prediction Button
# ------------------------------
st.divider()
model_choice = st.selectbox("Choose a Model", ["Logistic Regression", "Decision Tree"])

if st.button("Predict Cancellation"):
    
    # Logistic Regression requires scaling
    if model_choice == "Logistic Regression":
        input_scaled = scaler.transform(input_encoded)
        pred = lr_model.predict(input_scaled)[0]
    
    # Decision Tree does NOT require scaling
    else:
        pred = dt_model.predict(input_encoded)[0]

    # Output result
    if pred == 1:
        st.error("Prediction: ❌ Booking Will Be Canceled")
    else:
        st.success("Prediction: ✅ Booking Will Not Be Canceled")

import streamlit as st
import pandas as pd
import joblib

# App title
st.title("📱 Customer Churn Prediction")

# Load the model (with error handling)
try:
    model = joblib.load("expresso_model.pkl")
    st.write("✅ Model loaded successfully!")
except:
    st.error("❌ Could not load model file. Make sure 'expresso_model.pkl' is in the same folder.")
    st.stop()  # Stop the app if model can't be loaded

# Get the feature names that the model expects
feature_names = model.feature_names_in_
st.write(f"This model needs {len(feature_names)} features:")

# Show all the features the model needs
with st.expander("Click to see all required features"):
    for i, feature in enumerate(feature_names, 1):
        st.write(f"{i}. {feature}")

st.write("---")  # Add a line separator

# Collect user inputs
st.subheader("Enter Customer Data:")

# Create a dictionary to store user inputs
user_data = {}

# Ask user for input for each feature
for feature in feature_names:
    user_data[feature] = st.number_input(
        f"Enter {feature}:", 
        value=0.0,  # Default value
        step=0.01   # Allow decimal values
    )

# Add some space
st.write("")

# Prediction button
if st.button("🔮 Predict if Customer will Churn"):
    
    # Convert user inputs to a DataFrame (this is what the model expects)
    input_data = pd.DataFrame([user_data])
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        
        # Show result
        if prediction == 1:
            st.error("⚠️ Warning: This customer is likely to CHURN!")
        else:
            st.success("✅ Good news: This customer is likely to STAY!")
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.write("Please make sure all fields are filled correctly.")

# Add instructions for users
st.write("---")
st.subheader("ℹ️ How to use this app:")
st.write("1. Fill in all the customer data fields above")
st.write("2. Click the 'Predict' button")
st.write("3. See if the customer is likely to churn (leave) or stay")
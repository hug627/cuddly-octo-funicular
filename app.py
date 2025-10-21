import streamlit as st
import pandas as pd
import joblib
import numpy as np

# App title
st.title("📱 Customer Churn Prediction")
st.markdown("**Predict if a customer will churn based on their usage patterns**")

# Load the model (with error handling)
try:
    model = joblib.load("expresso_model.pkl")
    st.success("✅ Model loaded successfully!")
except:
    st.error("❌ Could not load model file. Make sure 'expresso_model.pkl' is in the same folder.")
    st.stop()  # Stop the app if model can't be loaded

# Get the feature names that the model expects
feature_names = model.feature_names_in_
st.write(f"**Model requires {len(feature_names)} features**")

# Show all the features the model needs
with st.expander("📋 Click to see all required features"):
    for i, feature in enumerate(feature_names, 1):
        st.write(f"{i}. {feature}")

st.write("---")  # Add a line separator

# Collect user inputs
st.subheader("📊 Enter Customer Data:")

# Create a dictionary to store user inputs
user_data = {}

# Group features by type for better organization
numeric_features = []
categorical_features = []

for feature in feature_names:
    if any(x in feature.lower() for x in ['montant', 'revenue', 'frequence', 'data_volume', 'on_net', 'orange', 'tigo', 'zone', 'regularity', 'freq']):
        numeric_features.append(feature)
    else:
        categorical_features.append(feature)

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("**💰 Financial & Usage Data:**")
    for feature in numeric_features[:len(numeric_features)//2]:
        if 'montant' in feature.lower() or 'revenue' in feature.lower():
            user_data[feature] = st.number_input(
                f"💵 {feature}:", 
                value=1000.0,
                step=100.0,
                help="Amount in local currency"
            )
        elif 'frequence' in feature.lower():
            user_data[feature] = st.number_input(
                f"📞 {feature}:", 
                value=10.0,
                step=1.0,
                help="Frequency of usage"
            )
        else:
            user_data[feature] = st.number_input(
                f"📊 {feature}:", 
                value=0.0,
                step=1.0
            )

with col2:
    st.markdown("**📱 Service & Network Data:**")
    for feature in numeric_features[len(numeric_features)//2:]:
        user_data[feature] = st.number_input(
            f"📊 {feature}:", 
            value=0.0,
            step=1.0
        )

# Handle categorical features (one-hot encoded)
st.markdown("**🏷️ Service Categories (0 or 1):**")
for feature in categorical_features:
    user_data[feature] = st.selectbox(
        f"🔘 {feature}:",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

# Add some space
st.write("")

# Prediction button
if st.button("🔮 Predict Customer Churn", type="primary"):
    
    # Convert user inputs to a DataFrame (this is what the model expects)
    input_data = pd.DataFrame([user_data])
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Show result with probability
        if prediction == 1:
            st.error(f"⚠️ **WARNING: This customer is likely to CHURN!**")
            st.write(f"Churn probability: {prediction_proba[1]:.1%}")
        else:
            st.success(f"✅ **Good news: This customer is likely to STAY!**")
            st.write(f"Retention probability: {prediction_proba[0]:.1%}")
            
        # Show confidence level
        confidence = max(prediction_proba)
        if confidence > 0.8:
            st.success(f"🎯 High confidence prediction ({confidence:.1%})")
        elif confidence > 0.6:
            st.warning(f"⚠️ Medium confidence prediction ({confidence:.1%})")
        else:
            st.error(f"❓ Low confidence prediction ({confidence:.1%})")
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.write("Please make sure all fields are filled correctly.")

# Add instructions for users
st.write("---")
st.subheader("ℹ️ How to use this app:")
st.write("1. **Fill in customer data** - Use realistic values based on customer behavior")
st.write("2. **Click Predict** - The model will analyze the data")
st.write("3. **Review results** - See churn probability and confidence level")
st.write("4. **Take action** - Use insights to retain at-risk customers")

# Add some sample data suggestions
with st.expander("💡 Sample Data Suggestions"):
    st.write("**Typical customer values:**")
    st.write("- MONTANT: 1000-10000 (amount spent)")
    st.write("- FREQUENCE_RECH: 5-20 (recharge frequency)")
    st.write("- REVENUE: 1000-15000 (monthly revenue)")
    st.write("- FREQUENCE: 10-30 (call frequency)")
    st.write("- REGULARITY: 20-60 (regularity score)")
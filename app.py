import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(page_title="Real Estate What-If Analyzer", layout="wide")

# Title
st.title("🏡 Real Estate What‑If Market Analyzer")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/housing.csv")

data = load_data()

# Sidebar inputs
st.sidebar.header("Property Features")

sqft = st.sidebar.slider("Square Footage", 500, 4000, 1500)
bedrooms = st.sidebar.slider("Bedrooms", 1, 6, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
age = st.sidebar.slider("House Age (Years)", 0, 100, 20)

# Select features and target
features = data[['sqft', 'bedrooms', 'bathrooms', 'age']]
target = data['price']

# Train model
model = LinearRegression()
model.fit(features, target)

# Prediction
prediction = model.predict([[sqft, bedrooms, bathrooms, age]])[0]

# Display prediction
st.subheader("💰 Estimated Property Price")
st.success(f"₹ {prediction:,.0f}")

# Visualizations
st.subheader("📊 Market Analysis")

col1, col2 = st.columns(2)

with col1:
    st.scatter_chart(data, x="sqft", y="price")

with col2:
    st.scatter_chart(data, x="age", y="price")

st.caption("Dataset used for academic purposes only")

import streamlit as st
from api_client import check_api_health, get_prediction

st.set_page_config(page_title="WA Housing Stress Dashboard", layout="wide")

st.title("WA Housing Model 2021")

# Sidebar - API Status
st.sidebar.header("System Status")
if check_api_health():
    st.sidebar.success("API Service: Online")
else:
    st.sidebar.error("API Service: Offline")

st.write("Welcome to the Housing Stress Simulator. Use the sidebar to adjust parameters.")

# Simple Test Button
if st.button("Run Test Prediction"):
    test_data = {
        "avg_weekly_income": 1800,
        "avg_weekly_mortgage": 500,
        "avg_weekly_rent": 400,
        "mining_concentration_ratio": 0.05,
        "unemployment_rate": 4.2,
        "avg_household_size": 2.5,
        "income_rent_gap": 1400
    }
    
    result = get_prediction(test_data)
    
    if result.get("status") == "success":
        st.metric("Predicted Housing Stress Index", f"{result['predicted_housing_stress_index']}%")
    else:
        st.error(f"Prediction failed: {result.get('message')}")
import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import json
from api_client import check_api_health, get_prediction

st.set_page_config(page_title="WA Housing Stress Dashboard", layout="wide")

# Geospatial Data Loading
@st.cache_data
def load_geospatial_data():
    # Read GPKG from data folder
    gdf = gpd.read_file("data/wa_census_spatial_2021.gpkg")
    
    # Standardise CRS for web mapping
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
        
    return json.loads(gdf.to_json())

# Master CSV Loading
@st.cache_data
def load_csv_data():
    # Read master CSV for tabular lookups
    return pd.read_csv("data/wa_census_master_2021.csv")

# Initialise data
try:
    geojson_data = load_geospatial_data()
    df_master = load_csv_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    geojson_data = None
    df_master = None

# Sidebar and Layout
st.title("WA Housing Model 2021")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Controls")
    if check_api_health():
        st.success("API Service: Online")
    else:
        st.error("API Service: Offline")
    
    # Map Legend
    st.subheader("Stress Index Legend")
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; font-size: 0.8em;">
            <span>Low (0%)</span>
            <span>High (100%)</span>
        </div>
        <div style="height: 20px; width: 100%; background: linear-gradient(to right, #ffff00, #ff0000); border-radius: 5px; margin-bottom: 10px;"></div>
        """,
        unsafe_allow_html=True
    )
    
    st.info("Use the simulator controls here to adjust demographic features.")

    # Connection Test
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
            st.metric("Predicted Stress Index", f"{result['predicted_housing_stress_index']}%")

# Map Section
with col2:
    st.subheader("Geospatial Risk Distribution")
    st.write("This map visualises the Housing Stress Index across Western Australia based on 2021 Census data. Darker red areas indicate higher financial pressure on households.")
    
    if geojson_data:
        # Perth coordinates
        view_state = pdk.ViewState(
            latitude=-31.9505,
            longitude=115.8605,
            zoom=9,
            pitch=0
        )

        # Pydeck layer configuration
        layer = pdk.Layer(
            "GeoJsonLayer",
            geojson_data,
            opacity=0.6,
            stroked=True,
            filled=True,
            get_fill_color="[255, (1 - properties.housing_stress_index / 100) * 255, 0, 150]",
            get_line_color=[255, 255, 255],
            pickable=True
        )

        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "Suburb: {SAL_NAME21}\nStress Index: {housing_stress_index}%"}
        ))
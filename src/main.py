import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import json
from api_client import check_api_health, get_prediction

st.set_page_config(page_title="WA Housing Stress Dashboard", layout="wide")

# Geospatial Data Loading
@st.cache_data
def load_data():
    gdf = gpd.read_file("data/wa_census_spatial_2021.gpkg")
    
    # Project to meters to calculate accurate centroids, then back to lat/lon
    projected_gdf = gdf.to_crs("EPSG:3857")
    centroids = projected_gdf.geometry.centroid.to_crs("EPSG:4326")
    gdf['centroid_lat'] = centroids.y
    gdf['centroid_lon'] = centroids.x
    
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Read CSV for full demographic features
    df = pd.read_csv("data/wa_census_master_2021.csv")
    
    # Dynamically find the Code and Name columns in the GPKG
    gpkg_cols = gdf.columns.tolist()
    code_col = next((c for c in gpkg_cols if 'CODE' in c.upper()), None)
    name_col = next((c for c in gpkg_cols if 'NAME' in c.upper()), None)
    
    if code_col and name_col:
        # Create a lookup andise types for matching
        names_lookup = gdf[[code_col, name_col, 'centroid_lat', 'centroid_lon']].copy()
        names_lookup[code_col] = names_lookup[code_col].astype(str)
        df['SAL_CODE_2021'] = df['SAL_CODE_2021'].astype(str)
        
        # Merge names into the CSV data
        df = df.merge(
            names_lookup, 
            left_on='SAL_CODE_2021', 
            right_on=code_col, 
            how='left'
        )
        # Standardise name column for the app
        df = df.rename(columns={name_col: 'DISPLAY_NAME'})
    
    return json.loads(gdf.to_json()), df

# Initialise data
try:
    geojson_data, df_master = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    geojson_data, df_master = None, None

# Sidebar and Layout
st.title("WA Housing Model 2021")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Controls")
    if check_api_health():
        st.success("API Service: Online")
    else:
        st.error("API Service: Offline")
    
    # Area Selector
    st.subheader("Select Location")
    if df_master is not None and 'DISPLAY_NAME' in df_master.columns:
        valid_df = df_master.dropna(subset=['DISPLAY_NAME'])
        area_list = sorted(valid_df['DISPLAY_NAME'].unique())
        
        # Default to Show All WA
        options = ["Show All WA"] + area_list
        selected_option = st.selectbox("Choose an Area", options)
        
        if selected_option != "Show All WA":
            area_data = valid_df[valid_df['DISPLAY_NAME'] == selected_option].iloc[0]
            target_lat = area_data['centroid_lat']
            target_lon = area_data['centroid_lon']
            target_zoom = 12
        else:
            # Perth default view
            target_lat = -31.9505
            target_lon = 115.8605
            target_zoom = 9
    else:
        st.warning("Area names could not be linked.")
        selected_option = "None"

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
    
    st.info("Adjust the sliders below (coming soon) to simulate economic changes in this Area.")

# Right Side Content
with col2:
    st.subheader("Geospatial Risk Distribution")
    st.write("This map visualises the Housing Stress Index. Darker red areas indicate higher financial pressure. Selected areas are highlighted in blue.")
    
    if geojson_data:
        # Base Heatmap Layer
        base_layer = pdk.Layer(
            "GeoJsonLayer",
            geojson_data,
            opacity=0.6,
            stroked=True,
            filled=True,
            get_fill_color="[255, (1 - properties.housing_stress_index / 100) * 255, 0, 150]",
            get_line_color=[255, 255, 255],
            pickable=True
        )

        layers = [base_layer]

        # Selection Highlight Layer
        if selected_option != "Show All WA":
            selected_geojson = {
                "type": "FeatureCollection",
                "features": [f for f in geojson_data['features'] if f['properties'].get('SAL_NAME21') == selected_option]
            }
            highlight_layer = pdk.Layer(
                "GeoJsonLayer",
                selected_geojson,
                opacity=0.8,
                stroked=True,
                filled=True,
                get_fill_color=[0, 150, 255, 180],
                get_line_color=[0, 255, 255],
                line_width_min_pixels=3
            )
            layers.append(highlight_layer)

        view_state = pdk.ViewState(
            latitude=target_lat,
            longitude=target_lon,
            zoom=target_zoom,
            pitch=0
        )

        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "Area: {SAL_NAME21}\nStress Index: {housing_stress_index}%"}
        ))

    st.divider()

    # Metrics Section moved below map
    if df_master is not None and selected_option != "Show All WA" and selected_option != "None":
        st.subheader(f"Baseline Metrics: {selected_option}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Weekly Income", f"${area_data['avg_weekly_income']:,.0f}")
        m2.metric("Weekly Rent", f"${area_data['avg_weekly_rent']:,.0f}")
        m3.metric("Weekly Mortgage", f"${area_data['avg_weekly_mortgage']:,.0f}")
        m4.metric("Unemployment", f"{area_data['unemployment_rate']:.1f}%")
        m5.metric("Stress Index", f"{area_data['housing_stress_index']:.1f}%")
    else:
        st.subheader("Baseline Metrics: Western Australia")
        st.write("Please select a specific Area from the sidebar to view detailed baseline statistics.")
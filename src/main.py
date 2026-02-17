import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import json
from api_client import check_api_health, get_prediction

st.set_page_config(page_title="WA Housing Model 2021", layout="wide")

# Custom CSS for the legend and UI
st.markdown("""
    <style>
    .legend-bar {
        height: 10px;
        width: 100%;
        background: linear-gradient(to right, #00ff00, #ffff00, #ff0000);
        border-radius: 5px;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    gdf = gpd.read_file("data/wa_census_spatial_2021.gpkg").to_crs("EPSG:4326")
    df_raw = pd.read_csv("data/wa_census_master_2021.csv")
    
    # Identify the exact 644 model features by excluding ID and Target
    target_col = 'housing_stress_index'
    id_col = 'SAL_CODE_2021'
    model_features = [c for c in df_raw.columns if c not in [target_col, id_col]]
    
    # Math in meters for centroids to avoid the warning, then back to degrees
    projected_gdf = gdf.to_crs(epsg=3857)
    centroids_geo = projected_gdf.geometry.centroid.to_crs(epsg=4326)
    gdf['centroid_lat'] = centroids_geo.y
    gdf['centroid_lon'] = centroids_geo.x
    
    code_col = 'SAL_CODE21' if 'SAL_CODE21' in gdf.columns else gdf.columns[0]
    name_col = 'SAL_NAME21' if 'SAL_NAME21' in gdf.columns else gdf.columns[1]
    
    names_lookup = gdf[[code_col, name_col, 'centroid_lat', 'centroid_lon']].copy()
    names_lookup[code_col] = names_lookup[code_col].astype(str)
    df_raw[id_col] = df_raw[id_col].astype(str)
    
    df = df_raw.merge(names_lookup, left_on=id_col, right_on=code_col, how='left')
    df = df.rename(columns={name_col: 'DISPLAY_NAME'})
    
    return json.loads(gdf.to_json()), df, model_features

geojson_data, df_master, model_keys = load_data()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Global Controls")
    is_online = check_api_health()
    st.info("API Status: Online" if is_online else "API Status: Offline")
    
    st.divider()
    valid_df = df_master.dropna(subset=['DISPLAY_NAME'])
    selected_option = st.selectbox("Choose an Area", ["Show All WA"] + sorted(valid_df['DISPLAY_NAME'].unique()))
    
    # Restore the Legend
    st.write("Housing Stress Legend")
    st.markdown('<div class="legend-bar"></div>', unsafe_allow_html=True)
    st.caption("Low (0%) <----------------------> High (100%)")
    
    st.divider()
    with st.form("simulator_form"):
        st.subheader("Scenario Simulator")
        inc_adj = st.slider("Weekly Income Adjustment ($)", -500, 500, 0, step=50)
        rent_adj = st.slider("Weekly Rent Adjustment ($)", -200, 200, 0, step=20)
        mort_adj = st.slider("Weekly Mortgage Adjustment ($)", -200, 200, 0, step=20)
        unemp_adj = st.slider("Unemployment Change (%)", -5.0, 10.0, 0.0, step=0.5)
        # Restore Mining
        mining_adj = st.slider("Mining Workforce Change (%)", -10.0, 10.0, 0.0, step=1.0)
        
        run_calc = st.form_submit_button("Calculate Prediction", type="primary", use_container_width=True)

# --- MAIN CONTENT ---
st.title("WA Housing Model 2021")
tab1, tab2, tab3 = st.tabs(["🗺️ Geospatial View", "📊 Economic Data", "📈 Prediction Results"])

with tab1:
    view_state = pdk.ViewState(latitude=-31.95, longitude=115.86, zoom=6)
    
    # Blue Highlighting Logic
    highlight_color = [0, 0, 255, 200] # Blue
    default_line = [255, 255, 255]      # White
    
    if selected_option != "Show All WA":
        area_data = valid_df[valid_df['DISPLAY_NAME'] == selected_option].iloc[0]
        view_state = pdk.ViewState(latitude=area_data['centroid_lat'], longitude=area_data['centroid_lon'], zoom=11)

    layers = [
        pdk.Layer(
            "GeoJsonLayer",
            geojson_data,
            opacity=0.4,
            stroked=True,
            filled=True,
            get_fill_color="[255, (1 - properties.housing_stress_index / 100) * 255, 0, 150]",
            # Highlight selected area in blue, others white
            get_line_color=f"properties.SAL_NAME21 == '{selected_option}' ? {highlight_color} : {default_line}",
            line_width_min_pixels=2 if selected_option != "Show All WA" else 1,
            pickable=True
        )
    ]
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state))

if selected_option != "Show All WA":
    area_row = valid_df[valid_df['DISPLAY_NAME'] == selected_option].iloc[0]
    
    with tab2:
        st.subheader(f"Baseline Data: {selected_option}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Weekly Income", f"${area_row['avg_weekly_income']:,.0f}")
        c2.metric("Weekly Rent", f"${area_row['avg_weekly_rent']:,.0f}")
        c3.metric("Stress Index", f"{area_row['housing_stress_index']:.1f}%")
        c4.metric("Mining Workforce", f"{area_row['total_mining']:.0f}")

    with tab3:
        if run_calc:
            with st.spinner('Analysing Scenario...'):
                # 1. Create a workspace dict
                sim_data = area_row.to_dict()
                
                # 2. Apply adjustments
                sim_data['avg_weekly_income'] += inc_adj
                sim_data['avg_weekly_rent'] += rent_adj
                sim_data['avg_weekly_mortgage'] += mort_adj
                sim_data['unemployment_rate'] += unemp_adj
                sim_data['total_mining'] = max(0, sim_data['total_mining'] * (1 + mining_adj/100))

                # 3. CONSTRUCT PAYLOAD: Strict 644 count
                # We only take keys that were in the original CSV model_features
                payload = {k: float(sim_data[k]) for k in model_keys}

                prediction = get_prediction(payload)
                
                if isinstance(prediction, dict) and 'housing_stress_index' in prediction:
                    new_val = prediction['housing_stress_index']
                    st.success("Analysis Complete")
                    st.metric("Predicted Stress Level", f"{new_val:.2f}%", 
                              delta=f"{new_val - area_row['housing_stress_index']:.2f}%", 
                              delta_color="inverse")
                else:
                    st.error(f"Payload error: Sent {len(payload)} features. Expected 644.")
                    with st.expander("Debug: Payload Verification"):
                        st.json(payload)
import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import json
import plotly.express as px
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
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Load raw data
    gdf = gpd.read_file("data/wa_census_spatial_2021.gpkg").to_crs("EPSG:4326")
    df_raw = pd.read_csv("data/wa_census_master_2021.csv")
    
    # --- GEOMETRY SIMPLIFICATION ---
    # Temporarily project to metres for accurate simplification, then back to degrees
    # Reduces number of polygon vertices to optimise browser performance.
    gdf = gdf.to_crs(epsg=3857)
    # Reduced tolerance to 100m to restore visual detail while maintaining performance
    gdf['geometry'] = gdf.geometry.simplify(tolerance=100, preserve_topology=True)
    gdf = gdf.to_crs(epsg=4326)
    
    # Identify model features STRICTLY from the raw CSV to avoid merge duplicates
    target_col = 'housing_stress_index'
    id_col = 'SAL_CODE_2021'
    model_features = [c for c in df_raw.columns if c not in [target_col, id_col]]
    
    # Centroid calculation for map snapping
    projected_gdf = gdf.to_crs(epsg=3857)
    centroids_geo = projected_gdf.geometry.centroid.to_crs(epsg=4326)
    gdf['centroid_lat'] = centroids_geo.y
    gdf['centroid_lon'] = centroids_geo.x
    
    code_col = 'SAL_CODE21' if 'SAL_CODE21' in gdf.columns else gdf.columns[0]
    name_col = 'SAL_NAME21' if 'SAL_NAME21' in gdf.columns else gdf.columns[1]
    
    # Prepare lookup
    names_lookup = gdf[[code_col, name_col, 'centroid_lat', 'centroid_lon']].copy()
    names_lookup[code_col] = names_lookup[code_col].astype(str)
    df_raw[id_col] = df_raw[id_col].astype(str)
    
    # Merge for display, but we keep model_features list clean from the original df_raw
    df = df_raw.merge(names_lookup, left_on=id_col, right_on=code_col, how='left')
    df = df.rename(columns={name_col: 'DISPLAY_NAME'})
    
    return json.loads(gdf.to_json()), df, model_features

geojson_data, df_master, model_keys = load_data()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Global Controls")
    is_online = check_api_health()
    if is_online:
        st.success("API Status: Online")
    else:
        st.error("API Status: Offline")
    
    st.divider()
    valid_df = df_master.dropna(subset=['DISPLAY_NAME'])
    selected_option = st.selectbox("Choose an Area", ["Show All WA"] + sorted(valid_df['DISPLAY_NAME'].unique()))
    
    # Legend
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
        mining_adj = st.slider("Mining Workforce Change (%)", -10.0, 10.0, 0.0, step=1.0)
        
        run_calc = st.form_submit_button("Calculate Prediction", type="primary", use_container_width=True)

# --- MAIN CONTENT ---
st.title("WA Housing Model 2021")
tab1, tab2, tab3 = st.tabs(["🗺️ Geospatial View", "📊 Area Metrics", "📈 Prediction Results"])

with tab1:
    # Default view (Perth)
    view_state = pdk.ViewState(latitude=-31.95, longitude=115.86, zoom=6)
    
    # Visual distinctness for selection
    highlight_color = [0, 255, 255, 255] # Bright Cyan
    
    if selected_option != "Show All WA":
        area_data = valid_df[valid_df['DISPLAY_NAME'] == selected_option].iloc[0]
        view_state = pdk.ViewState(
            latitude=area_data['centroid_lat'], 
            longitude=area_data['centroid_lon'], 
            zoom=12,
            pitch=45
        )

    layers = [
        pdk.Layer(
            "GeoJsonLayer",
            geojson_data,
            opacity=0.4,
            stroked=True,
            filled=True,
            get_fill_color="[255, (1 - properties.housing_stress_index / 100) * 255, 0, 150]",
            # Highlight selected area in Cyan, others faint white
            get_line_color=f"properties.SAL_NAME21 == '{selected_option}' ? {highlight_color} : [255, 255, 255, 50]",
            line_width_min_pixels=6 if selected_option != "Show All WA" else 1,
            pickable=True
        )
    ]
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state))

if selected_option != "Show All WA":
    area_row = valid_df[valid_df['DISPLAY_NAME'] == selected_option].iloc[0]
    
    with tab2:
        st.subheader(f"Baseline Stats: {selected_option}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Avg. Weekly Income", f"${area_row['avg_weekly_income']:,.0f}")
        c2.metric("Avg. Weekly Rent", f"${area_row['avg_weekly_rent']:,.0f}")
        c3.metric("Avg. Weekly Mortgage", f"${area_row['avg_weekly_mortgage']:,.0f}")
        c4.metric("Modelled Housing Stress", f"{area_row['housing_stress_index']:.1f}%")
        c5.metric("Mining Jobs", f"{area_row['total_mining']:.0f}")

        # Charts
        st.divider()
        col_left, col_mid, col_right = st.columns(3)
        
        with col_left:
            st.write("Employment Breakdown")
            emp_data = pd.DataFrame({
                'Status': ['Full Time', 'Part Time', 'Unemployed'],
                'Count': [area_row['total_employed_full_time'], area_row['total_employed_part_time'], area_row['total_unemployed']]
            })
            st.bar_chart(emp_data.set_index('Status'), color="#99d5ff", x_label="Employment Status", y_label="No. of People")
            
        with col_mid:
            st.write("Housing Tenure Mix")
            tenure_data = pd.DataFrame({
                'Type': ['Renting', 'Mortgage'],
                'Count': [area_row['renting_households_count'], area_row['mortgage_households_count']]
            })
            # Right chart updated to light red
            st.bar_chart(tenure_data.set_index('Type'), color="#ff9999", x_label="Tenure Type", y_label="No. of Households")

        with col_right:
            st.write("Mining vs Non-Mining Workforce")
            total_workforce = area_row['total_employed_full_time'] + area_row['total_employed_part_time']
            mining_jobs = area_row['total_mining']
            non_mining = max(0, total_workforce - mining_jobs)
            
            # Use Plotly Pie to fix the "empty chart until maximized" rendering bug
            fig = px.pie(
                values=[mining_jobs, non_mining],
                names=['Mining', 'Non-Mining'],
                hole=0.4,
                color_discrete_sequence=["#6cd48c", "#bd8112"]
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True, key=f"pie_{selected_option}")

    with tab3:
        if run_calc:
            if not is_online:
                st.error("Cannot calculate: API is offline.")
            else:
                with st.spinner('Calculating Scenario Impacts...'):
                    # Workspace
                    sim_data = area_row.to_dict()
                    
                    # Adjustments
                    sim_data['avg_weekly_income'] += inc_adj
                    sim_data['avg_weekly_rent'] += rent_adj
                    sim_data['avg_weekly_mortgage'] += mort_adj
                    sim_data['unemployment_rate'] = max(0, sim_data['unemployment_rate'] + unemp_adj)
                    sim_data['total_mining'] = max(0, sim_data['total_mining'] * (1 + mining_adj/100))

                    # Payload: Strictly filter to the 644 model_keys
                    payload = {k: float(sim_data[k]) for k in model_keys if k in sim_data}

                    prediction = get_prediction(payload)
                    
                    # Check for the correct key from Flask: 'predicted_housing_stress_index'
                    if isinstance(prediction, dict) and 'predicted_housing_stress_index' in prediction:
                        new_val = prediction['predicted_housing_stress_index']
                        st.balloons()
                        st.success("Simulation Complete")
                        
                        curr_val = area_row['housing_stress_index']
                        st.metric(
                            label="Predicted Housing Stress Index", 
                            value=f"{new_val:.2f}%", 
                            delta=f"{new_val - curr_val:.2f}%", 
                            delta_color="inverse"
                        )
                        
                        # Comparison display
                        st.write(f"The simulated changes result in a **{abs(new_val - curr_val):.2f}%** {'increase' if new_val > curr_val else 'decrease'} in housing stress for {selected_option}.")
                    else:
                        st.error(f"Prediction Failed. Sent {len(payload)} features.")
                        with st.expander("Debug Info"):
                            st.write("API Response:", prediction)
                            st.write("Features Sent:", list(payload.keys()))
else:
    st.info("Select a specific area from the sidebar to enable the Area Metrics and Scenario Simulator.")
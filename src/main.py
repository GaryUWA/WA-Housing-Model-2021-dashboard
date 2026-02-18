import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import json
import plotly.express as px
import joblib
import os
import sklearn 
from api_client import check_api_health, get_prediction

st.set_page_config(page_title="WA Housing Model 2021", layout="wide")

# Initialise session state for connection mode
if 'use_local' not in st.session_state:
    st.session_state.use_local = False

# Custom CSS for the legend, metrics, and scrollbar visibility
st.markdown("""
    <style>
    /* Legend styling */
    .legend-bar {
        height: 10px;
        width: 100%;
        background: linear-gradient(to right, #00ff00, #ffff00, #ff0000);
        border-radius: 5px;
        margin-top: 5px;
    }
    /* Metric card styling */
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    
    /* Force sidebar and dropdown menu scrollbars to be visible and grabbable */
    section[data-testid="stSidebar"] .st-emotion-cache-6qob1r, 
    ul[data-testid="stSelectboxVirtualList"] {
        overflow-y: auto !important;
    }

    /* Target both sidebar and the dropdown popover scrollbars */
    section[data-testid="stSidebar"] ::-webkit-scrollbar,
    [data-testid="stVirtualList"] ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
        display: block !important;
    }

    section[data-testid="stSidebar"] ::-webkit-scrollbar-track,
    [data-testid="stVirtualList"] ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }

    section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb,
    [data-testid="stVirtualList"] ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
        border: 2px solid #f1f1f1;
    }

    section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover,
    [data-testid="stVirtualList"] ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_local_models():
    """Load joblib models for local inference fallback."""
    # Ensure paths are correct relative to root
    model = joblib.load("models/housing_stress_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    return model, scaler

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
    
    # Identify model features
    # Retrieve from scaler to ensure exact match with training phase
    _, scaler = load_local_models()
    model_features = list(scaler.feature_names_in_)
    
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
    
    target_col = 'housing_stress_index'
    id_col = 'SAL_CODE_2021'
    df_raw[id_col] = df_raw[id_col].astype(str)
    
    # Merge for display, but we keep model_features list clean from the original df_raw
    df = df_raw.merge(names_lookup, left_on=id_col, right_on=code_col, how='left')
    df = df.rename(columns={name_col: 'DISPLAY_NAME'})

    # Pre-format map hover string specifically for tooltip
    gdf['stress_label'] = gdf['housing_stress_index'].apply(lambda x: f"{x:.2f}")
    
    return json.loads(gdf.to_json()), df, model_features

geojson_data, df_master, model_keys = load_data()

# Initialize session state for the confirmed selection
# This ensures "Show All WA" is the default on initial load.
if 'confirmed_selection' not in st.session_state:
    st.session_state.confirmed_selection = "Show All WA"

# --- SIDEBAR ---
with st.sidebar:
    st.header("Global Controls")
    
    # Connection Logic
    if st.session_state.use_local:
        st.warning("Mode: Local Inference")
        if st.button("Connect to API"):
            st.session_state.use_local = False
            st.rerun()
    else:
        is_online = check_api_health() # This has its own timeout in api_client
        if is_online:
            st.success("API Status: Online")
        else:
            st.error("API Status: Offline")
            st.session_state.use_local = True
            st.rerun()
    
    st.divider()
    valid_df = df_master.dropna(subset=['DISPLAY_NAME'])
    
    # --- AREA SELECTION FORM ---
    # wrapping this in a form prevents selectbox from triggering a rerun until button is hit
    with st.form("area_selection_form"):
        selected_option = st.selectbox(
            "Choose an Area", 
            ["Show All WA"] + sorted(valid_df['DISPLAY_NAME'].unique()),
            # Ensures dropdown visually reflects what is actually currently loaded
            index=(["Show All WA"] + sorted(valid_df['DISPLAY_NAME'].unique())).index(st.session_state.confirmed_selection)
        )

        # Button acts as the "Calculate" equivalent for the map and metrics
        if st.form_submit_button("Select Area", type="primary", use_container_width=True):
            st.session_state.confirmed_selection = selected_option
            st.rerun() 
    
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

# Use the confirmed selection for the rest of the app logic
active_area = st.session_state.confirmed_selection

# --- MAIN CONTENT ---
st.title("WA Housing Model 2021")
tab1, tab2, tab3 = st.tabs(["🗺️ Geospatial View", "📊 Area Metrics", "📈 Scenario Results"])

with tab1:
    # Default view (Perth)
    view_state = pdk.ViewState(latitude=-31.95, longitude=115.86, zoom=6)
    # Visual distinctness for selection
    highlight_color = [0, 255, 255, 255] 
    
    if active_area != "Show All WA":
        area_data = valid_df[valid_df['DISPLAY_NAME'] == active_area].iloc[0]
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
            get_line_color=f"properties.SAL_NAME21 == '{active_area}' ? {highlight_color} : [255, 255, 255, 50]",
            line_width_min_pixels=6 if active_area != "Show All WA" else 1,
            pickable=True
        )
    ]
    # Spinner for notifying that the geospatial layers are rendering
    with st.spinner("Loading..."):
        st.pydeck_chart(pdk.Deck(
            layers=layers, 
            initial_view_state=view_state,
            tooltip={"text": "Area: {SAL_NAME21}\nStress Index: {stress_label}%"}
        ))

if active_area != "Show All WA":
    area_row = valid_df[valid_df['DISPLAY_NAME'] == active_area].iloc[0]
    
    with tab2:
        st.subheader(f"Baseline Stats: {active_area}")
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
            
            fig.update_traces(hovertemplate="<b>Sector:</b> %{label}<br><b>Count:</b> %{value:,.0f}<extra></extra>")
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True, key=f"pie_{active_area}")

        # --- HOUSING STRESS DISTRIBUTION ---
        st.divider()
        st.subheader("State-wide Stress Distribution")
        st.write(f"How {active_area} compares to all other areas in Western Australia.")
        
        hist_fig = px.histogram(
            valid_df, 
            x="housing_stress_index",
            nbins=50,
            title="Distribution of Housing Stress Across WA",
            labels={'housing_stress_index': 'Housing Stress Index (%)'},
            color_discrete_sequence=['#cbd5e0']
        )

        # Add a vertical line for the selected area
        hist_fig.add_vline(
            x=area_row['housing_stress_index'], 
            line_width=3, 
            line_dash="dash", 
            line_color="#e53e3e",
            annotation_text=f"{active_area}: {area_row['housing_stress_index']:.1f}%",
            annotation_position="top right"
        )
        
        hist_fig.update_traces(hovertemplate="<b>Stress Level:</b> %{x:.1f}%<br><b>Areas Count:</b> %{y}<extra></extra>")
        hist_fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=350, yaxis_title="Number of Areas")
        st.plotly_chart(hist_fig, use_container_width=True)

    with tab3:
        if run_calc:
            with st.spinner('Calculating Scenario Impacts...'):
                # Prepare workspace
                sim_data = area_row.to_dict()
                
                # Apply Adjustments
                sim_data['avg_weekly_income'] += inc_adj
                sim_data['avg_weekly_rent'] += rent_adj
                sim_data['avg_weekly_mortgage'] += mort_adj
                sim_data['unemployment_rate'] = max(0, sim_data['unemployment_rate'] + unemp_adj)
                sim_data['total_mining'] = max(0, sim_data['total_mining'] * (1 + mining_adj/100))

                # Payload: Strictly filter to the model_keys
                payload = {k: float(sim_data[k]) for k in model_keys if k in sim_data}

                # --- PREDICTION LOGIC (API vs LOCAL) ---
                prediction_val = None
                
                if not st.session_state.use_local:
                    try:
                        # Attempt API call
                        resp = get_prediction(payload)
                        if resp and 'predicted_housing_stress_index' in resp:
                            prediction_val = resp['predicted_housing_stress_index']
                    except Exception:
                        st.session_state.use_local = True
                        st.warning("API connection failed during calculation. Falling back to local model.")

                # Fallback to local joblib if API failed or we are in local mode
                if prediction_val is None:
                    try:
                        model, scaler = load_local_models()
                        input_df = pd.DataFrame([payload])
                        # Reorder to match model training features
                        input_df = input_df[model_keys]
                        scaled_data = scaler.transform(input_df)
                        prediction_val = model.predict(scaled_data)[0]
                    except Exception as e:
                        st.error(f"Local inference failed: {e}")

                # --- DISPLAY RESULTS ---
                if prediction_val is not None:
                    st.balloons()
                    st.success(f"Simulation Complete ({'Local Mode' if st.session_state.use_local else 'Live API'})")
                    
                    curr_val = area_row['housing_stress_index']
                    st.metric(
                        label="Predicted Housing Stress Index", 
                        value=f"{prediction_val:.2f}%", 
                        delta=f"{prediction_val - curr_val:.2f}%", 
                        delta_color="inverse"
                    )
                    
                    st.write(f"The simulated changes result in a **{abs(prediction_val - curr_val):.2f}%** {'increase' if prediction_val > curr_val else 'decrease'} in housing stress for {active_area}.")
                    
                    # --- FEATURE IMPORTANCE PLOT ---
                    st.divider()
                    st.subheader("Scenario Change Intensity")
                    
                    impact_data = pd.DataFrame({
                        'Feature': ['Income', 'Rent', 'Mortgage', 'Unemployment', 'Mining'],
                        'Relative Change': [abs(inc_adj)/500, abs(rent_adj)/200, abs(mort_adj)/200, abs(unemp_adj)/10, abs(mining_adj)/10]
                    }).sort_values('Relative Change', ascending=True)

                    fig_imp = px.bar(
                        impact_data,
                        x='Relative Change',
                        y='Feature',
                        orientation='h',
                        title="Magnitude of Input Adjustments",
                        labels={'Relative Change': 'Normalised Adjustment Factor (0-1)'},
                        color='Relative Change',
                        color_continuous_scale='Blues'
                    )
                    fig_imp.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig_imp, use_container_width=True)

                    # --- SCENARIO COMPARISON TABLE ---
                    st.divider()
                    st.subheader("Scenario Comparison Detail")
                    
                    comparison_df = pd.DataFrame({
                        "Parameter": ["Avg. Weekly Income", "Avg. Weekly Rent", "Avg. Weekly Mortgage", "Unemployment Rate", "Mining Workforce"],
                        "Baseline": [
                            f"${area_row['avg_weekly_income']:,.0f}",
                            f"${area_row['avg_weekly_rent']:,.0f}",
                            f"${area_row['avg_weekly_mortgage']:,.0f}",
                            f"{area_row['unemployment_rate']:.1f}%",
                            f"{area_row['total_mining']:,.0f}"
                        ],
                        "Simulated": [
                            f"${sim_data['avg_weekly_income']:,.0f}",
                            f"${sim_data['avg_weekly_rent']:,.0f}",
                            f"${sim_data['avg_weekly_mortgage']:,.0f}",
                            f"{sim_data['unemployment_rate']:.1f}%",
                            f"{sim_data['total_mining']:,.0f}"
                        ],
                        "Change": [
                            f"{'+' if inc_adj > 0 else ''}${inc_adj:,.0f}",
                            f"{'+' if rent_adj > 0 else ''}${rent_adj:,.0f}",
                            f"{'+' if mort_adj > 0 else ''}${mort_adj:,.0f}",
                            f"{'+' if unemp_adj > 0 else ''}{unemp_adj:.1f}%",
                            f"{(mining_adj):.1f}% factor"
                        ]
                    })
                    st.table(comparison_df)
else:
    st.info("Select a specific area from the sidebar to enable the Area Metrics and Scenario Simulator.")
# WA-Housing-Model-2021-dashboard

## Project Overview
This repository contains the **Interactive Visualisation Layer** for the WA Housing Model Project. It provides a [Streamlit](https://streamlit.io) dashboard that allows users to simulate housing stress scenarios across Western Australia using an interactive map, dynamic demographic sliders, and real-time model inference.

This project demonstrates a modular machine learning architecture:
1. **`WA-Housing-Model-2021-analysis`**: Data ETL, Geospatial Wrangling, and Model Training.
2. **`WA-Housing-Model-2021-service`**: Dockerised Flask API for production-grade inference.
3. **`WA-Housing-Model-2021-dashboard` (Current Repo)**: Streamlit dashboard for visualisation and scenario simulation.

## Live Demo
The dashboard is deployed and accessible via **Streamlit Community Cloud**. This provides a hosted alternative to manual reproduction, allowing for immediate exploration of the model's capabilities:
**[View Live Dashboard](https://wa-housing-model-2021-dashboard-hnn3poaha5r7uhhgucuzjs.streamlit.app/)**

## App Structure & Features
The application is organised into three functional tabs and a central control sidebar:

* **Geospatial View (Tab 1):** A high-performance Mapbox-powered deck.gl map rendering WA Statistical Areas (SAL). It provides a choropleth visualisation of the Predicted Housing Stress Index across the state.
* **Area Metrics (Tab 2):** Detailed demographic breakdown and employment charts for specific locations. *Note: Requires selecting an area from the sidebar.*
* **Scenario Results (Tab 3):** Comparative analysis between baseline census data and the simulated scenario results. *Note: Requires a calculated scenario.*
* **Scenario Simulator (Sidebar):** Users can adjust key census variables (such as income, rent, and unemployment) to define a custom housing scenario. *Note: This requires selecting an area first to populate baseline data.* A dedicated **"Calculate Prediction"** button triggers the inference engine.
* **Inference Engine:** Features built-in API reconnectivity. The app prioritises the Dockerised Flask service but seamlessly falls back to local `.joblib` inference if the API is unreachable.

## Tech Stack
* **UI Framework:** [Streamlit](https://streamlit.io) (Python-native web apps)
* **Visualisation:** [Pydeck](https://deckgl.readthedocs.io/) (High-scale spatial rendering) & [Plotly](https://plotly.com/python/) (Interactive charting)
* **Spatial Data:** [Geopandas](https://geopandas.org/) (Geospatial engine)
* **ML Inference:** [Scikit-Learn](https://scikit-learn.org) & [Joblib](https://joblib.readthedocs.io)

## Getting started (reproducibility)

To get started, first clone the repository:
```bash
git clone https://github.com/GaryUWA/WA-Housing-Model-2021-dashboard.git
cd WA-Housing-Model-2021-dashboard
```

### Option 1: Running in Microservice Mode (Recommended)
This mode connects the dashboard to the [WA-Housing-Model-2021-service](https://github.com/GaryUWA/WA-Housing-Model-2021-service.git) API.

1.  **Setup Environment:**
    ```bash
    python -m venv .venv
    source .venv/scripts/activate # Windows
    # source .venv/bin/activate # macOS/Linux
    pip install -r requirements.txt
    ```

2.  **Configure Connection:** 
    ```bash
    cp .env.example .env
    # Edit .env to ensure API_URL matches your local service address
    ```

3.  **Start the Backend Service:** Ensure the Flask service from **Repo 2** is running via **Docker**.

4.  **Run the Dashboard:**
    ```bash
    streamlit run src/main.py
    ```

### Option 2: Running in Standalone Mode
If the Flask service is not active, the dashboard will automatically detect the connection failure and switch to **Local Inference Mode**. It will load the trained model directly from the `models/` directory using `joblib`. Follow **Option 1** but skip **Step 3**.

## Disclaimer
This project is for **educational and portfolio purposes only**. The predictions are based on independent machine learning models trained on 2021 ABS Census data and should not be used for real-world financial or policy decisions. Development was AI-assisted by Gemini for framework scaffolding, rapid development, and debugging; no sensitive information from the ABS datasets were shared with the AI during this process.

## Attribute & License
This project is under the [MIT License](./LICENSE)

## Author
* **Zain Zabidi** - Data Science Graduate (UWA)
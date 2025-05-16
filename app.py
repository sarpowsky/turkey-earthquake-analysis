# app.py
# Path: /app.py
import streamlit as st
import os
import pandas as pd

# Import utility functions
from utils.data_loader import load_earthquake_data, load_fault_data, load_model, get_city_risk_data

# Import UI components
from components.sidebar import create_sidebar
from components.home import show_home
from components.data_explorer import show_data_explorer
from components.map_viewer import show_map_viewer
from components.model_predictor import show_model_predictor
from components.clustering import show_clustering
from components.risk_assessment import show_risk_assessment

# Configure page
st.set_page_config(
    page_title="Turkish Earthquake Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data (only once)
@st.cache_data
def load_data():
    """Load all required data"""
    earthquake_df = load_earthquake_data()
    fault_gdf = load_fault_data()
    model, scaler = load_model()  # model will be a pipeline or a model based on what's available
    city_risk_df = get_city_risk_data()
    return earthquake_df, fault_gdf, model, scaler, city_risk_df

# Load data
earthquake_df, fault_gdf, model, scaler, city_risk_df = load_data()

# Create sidebar and get selection
selection = create_sidebar()

# Display the selected page
if selection == "Home":
    show_home()
elif selection == "Data Explorer":
    show_data_explorer(earthquake_df)
elif selection == "Map Viewer":
    show_map_viewer(earthquake_df, fault_gdf)
elif selection == "Magnitude Predictor":
    show_model_predictor(earthquake_df, model, scaler)
elif selection == "Cluster Analysis":
    show_clustering(earthquake_df)
elif selection == "Risk Assessment":
    show_risk_assessment(earthquake_df, fault_gdf, city_risk_df)
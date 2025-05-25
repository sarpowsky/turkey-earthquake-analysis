# components/map_viewer.py
# Path: /components/map_viewer.py
import streamlit as st
import pandas as pd
import os

def show_map_viewer(df, fault_gdf):
    """Interactive map visualization page using pre-rendered maps"""
    st.title("Earthquake Map Viewer")
    
    # Map type selection
    map_type = st.selectbox(
        "Select Map Type",
        ["Earthquake Distribution", "K-Means Clusters", "DBSCAN Clusters", "Risk Zones"]
    )
    
    # Map file paths
    map_files = {
        "Earthquake Distribution": "maps/enhanced_earthquake_map.html",
        "K-Means Clusters": "maps/enhanced_kmeans_clusters_map.html",
        "DBSCAN Clusters": "maps/enhanced_dbscan_clusters_map.html",
        "Risk Zones": "maps/enhanced_earthquake_risk_map.html"
    }
    
    # Display selected map description
    if map_type == "Earthquake Distribution":
        st.write("This map shows the distribution of earthquakes, with strong earthquakes (>6.0) marked in red.")
    elif map_type == "K-Means Clusters":
        st.write("This map shows the K-Means clustering results, grouping earthquakes into spatial patterns.")
    elif map_type == "DBSCAN Clusters":
        st.write("This map shows DBSCAN clustering results, identifying dense clusters and outliers.")
    elif map_type == "Risk Zones":
        st.write("This map shows earthquake risk zones based on magnitude, cluster density, and fault line proximity.")
    
    # Display the selected map
    file_path = map_files.get(map_type)
    
    if file_path and os.path.exists(file_path):
        # Load and display pre-rendered map
        st.components.v1.html(open(file_path, 'r', encoding='utf-8').read(), height=600)
    else:
        st.error(f"Pre-rendered map not found at {file_path}.")
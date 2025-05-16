# utils/map_utils.py
# Path: /utils/map_utils.py
import folium
from folium.plugins import HeatMap, MarkerCluster
import streamlit as st
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import branca.colormap as cm

def create_base_map(center=[38.5, 35.5], zoom=6):
    """Create a base map centered on Turkey"""
    m = folium.Map(location=center, zoom_start=zoom, tiles="cartodbpositron")
    return m

def add_fault_lines(m, fault_gdf, importance_threshold=0):
    """Add fault lines to the map"""
    if fault_gdf is None:
        return m
        
    # Filter faults by importance if desired
    if importance_threshold > 0:
        fault_data = fault_gdf[fault_gdf['importance'] >= importance_threshold]
    else:
        fault_data = fault_gdf
    
    # Color by importance
    def style_function(feature):
        importance = feature['properties']['importance']
        color = '#FF0000' if importance >= 4 else '#FFA500' if importance >= 3 else '#FFFF00'
        return {
            'color': color,
            'weight': importance * 0.5,  # Thicker lines for more important faults
            'opacity': 0.7
        }
    
    # Add GeoJSON to map
    folium.GeoJson(
        fault_data,
        name='Fault Lines',
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=['FAULT_NAME', 'importance']),
    ).add_to(m)
    
    return m

def create_earthquake_map(df, fault_gdf=None):
    """Create an earthquake distribution map"""
    # Create base map
    m = create_base_map()
    
    # Add heatmap layer
    heat_data = [[row['Latitude'], row['Longitude']] for _, row in df.sample(min(3000, len(df))).iterrows()]
    HeatMap(heat_data, radius=8, gradient={'0.4': 'blue', '0.6': 'cyan', '0.8': 'yellow', '1.0': 'red'}).add_to(m)
    
    # Add markers for strong earthquakes (magnitude > 6)
    for idx, row in df[df['Magnitude'] > 6].iterrows():
        popup_content = f"""
        <div style="font-family: Arial; min-width: 200px;">
            <h4 style="margin-bottom: 5px; color: #d32f2f;">Earthquake Details</h4>
            <b>Magnitude:</b> {row['Magnitude']:.1f}<br>
            <b>Depth:</b> {row['Depth']:.1f} km<br>
            <b>Location:</b> {row.get('Location', 'Unknown')}<br>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=row['Magnitude'] * 1.5,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup=folium.Popup(popup_content, max_width=300),
        ).add_to(m)
    
    # Add fault lines if available
    if fault_gdf is not None:
        m = add_fault_lines(m, fault_gdf, importance_threshold=3)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def create_cluster_map(df, cluster_column, fault_gdf=None):
    """Create a map showing clusters"""
    # Create base map
    m = create_base_map()
    
    # Create a discrete color map for clusters
    clusters = df[cluster_column].unique()
    cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'cadetblue']
    
    # Use all data points, no sampling
    for idx, row in df.iterrows():
        cluster_idx = int(row[cluster_column]) % len(cluster_colors) if row[cluster_column] >= 0 else -1
        color = cluster_colors[cluster_idx] if cluster_idx >= 0 else 'black'
        
        # Enhanced radius calculation from notebook for consistency
        if row['Magnitude'] >= 7:
            # Exponential growth for major earthquakes
            radius = 15 + ((row['Magnitude'] - 7) ** 2) * 6
        else:
            radius = 3 + (row['Magnitude'] - 4) ** 1.5  # Exponential growth
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7 if cluster_idx >= 0 else 0.3,  # Make noise points more transparent
            popup=f"Cluster: {row[cluster_column]}<br>Magnitude: {row['Magnitude']}"
        ).add_to(m)
    
    # Add fault lines if available
    if fault_gdf is not None:
        m = add_fault_lines(m, fault_gdf, importance_threshold=3)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def create_risk_map(df, fault_gdf=None):
    """Create a risk zone map"""
    # Create base map
    m = create_base_map()
    
    # Color mapping for risk zones
    risk_colors = {
        'Low': 'green',
        'Medium': 'orange',
        'High': 'red'
    }
    
    # Add markers colored by risk zone - use full dataset for better grouping
    for idx, row in df.iterrows():
        risk_zone = row.get('Risk_Zone', 'Low')
        color = risk_colors.get(risk_zone, 'green')
        
        # Enhanced radius calculation from notebook for consistency
        if row['Magnitude'] >= 7:
            # Exponential growth for major earthquakes
            radius = 15 + ((row['Magnitude'] - 7) ** 2) * 6
        else:
            radius = 3 + (row['Magnitude'] - 4) ** 1.5  # Exponential growth
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"Risk: {risk_zone}<br>Magnitude: {row['Magnitude']}"
        ).add_to(m)
    
    # Add fault lines if available
    if fault_gdf is not None:
        m = add_fault_lines(m, fault_gdf, importance_threshold=3)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def display_folium_map(m):
    """Display a folium map in Streamlit"""
    folium_static(m)
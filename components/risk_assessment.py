# components/risk_assessment.py
# Path: /components/risk_assessment.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os

def show_risk_assessment(df, fault_gdf, city_risk_df=None):
    """Risk assessment visualization page"""
    st.title("Earthquake Risk Assessment")
    
    # Check if risk data is available
    has_risk_data = 'Risk_Zone' in df.columns
    
    if not has_risk_data:
        st.warning("Risk zone data not available in the dataset. Showing limited information.")
    
    # Tabs for different risk visualizations
    tab1, tab2 = st.tabs(["City Risk Assessment", "Risk Zone Analysis"])
    
    with tab1:
        show_city_risk_analysis(df, fault_gdf, city_risk_df)

    with tab2:
        show_risk_zone_analysis(df, fault_gdf, has_risk_data)

def show_risk_zone_analysis(df, fault_gdf, has_risk_data):
    """Show risk zone analysis"""
    if has_risk_data:
        # Risk zone distribution
        st.write("### Risk Zone Distribution")
        risk_distribution = df['Risk_Zone'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        risk_distribution.plot(kind='bar', color=['green', 'yellow', 'red'], ax=ax)
        ax.set_title('Distribution of Earthquake Risk Zones')
        ax.set_xlabel('Risk Zone')
        ax.set_ylabel('Number of Earthquakes')
        st.pyplot(fig)
        
        # Risk zone characteristics
        st.write("### Risk Zone Characteristics")
        risk_analysis = df.groupby('Risk_Zone').agg({
            'Magnitude': ['mean', 'min', 'max', 'count'],
            'Depth': ['mean', 'min', 'max']
        })
        
        # Flatten MultiIndex columns
        risk_analysis.columns = ['_'.join(col).strip() for col in risk_analysis.columns.values]
        st.dataframe(risk_analysis.round(2))
    
    # Show pre-rendered risk heatmap
    st.write("### Earthquake Density Heatmap")
    st.write("Areas with higher earthquake density generally have higher seismic risk.")
    
    density_map_path = "maps/enhanced_earthquake_density_map.html"
    if os.path.exists(density_map_path):
        st.components.v1.html(open(density_map_path, 'r', encoding='utf-8').read(), height=600)
    else:
        st.error(f"Pre-rendered density map not found at {density_map_path}.")

def show_city_risk_analysis(df, fault_gdf, city_risk_df):
    """Show city risk analysis"""
    st.write("### City Earthquake Risk Assessment")
    
    # Create a list of major Turkish cities (needed for context)
    cities = [
        {"name": "Istanbul", "lat": 41.0082, "lon": 28.9784, "population": 15655924},
        {"name": "Ankara", "lat": 39.9334, "lon": 32.8597, "population": 5803482},
        {"name": "Izmir", "lat": 38.4237, "lon": 27.1428, "population": 4479525},
        {"name": "Bursa", "lat": 40.1885, "lon": 29.0610, "population": 3214571},
        {"name": "Antalya", "lat": 36.8969, "lon": 30.7133, "population": 2696249},
        {"name": "Adana", "lat": 37.0000, "lon": 35.3213, "population": 2270298},
        {"name": "Konya", "lat": 37.8667, "lon": 32.4833, "population": 2320241},
        {"name": "Gaziantep", "lat": 37.0662, "lon": 37.3833, "population": 2164134},
        {"name": "Şanlıurfa", "lat": 37.1591, "lon": 38.7969, "population": 2213964},
        {"name": "Mersin", "lat": 36.8000, "lon": 34.6333, "population": 1938389},
        {"name": "Diyarbakır", "lat": 37.9144, "lon": 40.2306, "population": 1818133},
        {"name": "Hatay", "lat": 36.2025, "lon": 36.1606, "population": 1544640},
        {"name": "Manisa", "lat": 38.6191, "lon": 27.4289, "population": 1475716},
        {"name": "Kayseri", "lat": 38.7205, "lon": 35.4826, "population": 1445683},
        {"name": "Samsun", "lat": 41.2867, "lon": 36.3300, "population": 1377546},
        {"name": "Balıkesir", "lat": 39.6484, "lon": 27.8826, "population": 1273519},
        {"name": "Kahramanmaraş", "lat": 37.5750, "lon": 36.9228, "population": 1116618},
        {"name": "Van", "lat": 38.4942, "lon": 43.3800, "population": 1127612},
        {"name": "Aydın", "lat": 37.8444, "lon": 27.8458, "population": 1161702},
        {"name": "Denizli", "lat": 37.7765, "lon": 29.0864, "population": 1061043},
        {"name": "Sakarya", "lat": 40.7731, "lon": 30.3943, "population": 1100747},
        {"name": "Tekirdağ", "lat": 40.9833, "lon": 27.5167, "population": 1167059},
        {"name": "Muğla", "lat": 37.2153, "lon": 28.3636, "population": 1066736},
        {"name": "Eskişehir", "lat": 39.7767, "lon": 30.5206, "population": 915418},
        {"name": "Mardin", "lat": 37.3212, "lon": 40.7245, "population": 888874},
        {"name": "Malatya", "lat": 38.3552, "lon": 38.3095, "population": 742725},
        {"name": "Trabzon", "lat": 41.0053, "lon": 39.7267, "population": 824352},
        {"name": "Erzurum", "lat": 39.9042, "lon": 41.2705, "population": 749993},
        {"name": "Ordu", "lat": 40.9839, "lon": 37.8764, "population": 775800},
        {"name": "Afyonkarahisar", "lat": 38.7587, "lon": 30.5387, "population": 751344},
        {"name": "Çanakkale", "lat": 40.1553, "lon": 26.4142, "population": 570499},
        {"name": "Düzce", "lat": 40.8438, "lon": 31.1565, "population": 409865},
        {"name": "Bingöl", "lat": 38.8855, "lon": 40.4983, "population": 285655},
        {"name": "Tokat", "lat": 40.3167, "lon": 36.5500, "population": 606934},
        {"name": "Kütahya", "lat": 39.4167, "lon": 29.9833, "population": 575671},
        {"name": "Batman", "lat": 37.8812, "lon": 41.1351, "population": 647205},
        {"name": "Elazığ", "lat": 38.6748, "lon": 39.2225, "population": 604411},
        {"name": "Çorum", "lat": 40.5489, "lon": 34.9533, "population": 528351},
        {"name": "Biga (Çanakkale)(Memleket <3)", "lat": 40.2322, "lon": 27.2464, "population": 87000},
        {"name": "Simav (Kütahya)", "lat": 39.0922, "lon": 28.9789, "population": 64000},
        {"name": "Göksun (Kahramanmaraş)", "lat": 38.0203, "lon": 36.4825, "population": 31000},
        {"name": "Kastamonu", "lat": 41.3887, "lon": 33.7827, "population": 383000},
        {"name": "Burdur", "lat": 37.7215, "lon": 30.2886, "population": 270000},
        {"name": "Kars", "lat": 40.6013, "lon": 43.0950, "population": 289000},
        {"name": "Adıyaman", "lat": 37.7636, "lon": 38.2773, "population": 632000},
        {"name": "Çankırı", "lat": 40.6013, "lon": 33.6134, "population": 195000},
        {"name": "Edirne", "lat": 41.6771, "lon": 26.5557, "population": 411000},
        {"name": "Bartın", "lat": 41.6358, "lon": 32.3375, "population": 198000},
        {"name": "Erzincan", "lat": 39.75, "lon": 39.49, "population": 234000},
        {"name": "Hakkari", "lat": 37.57, "lon": 43.74, "population": 267000},
        {"name": "Osmaniye", "lat": 37.0746, "lon": 36.2464, "population": 534000}
    ]
    
    if city_risk_df is None:
        # Calculate risk for each city
        city_risk_df = calculate_city_risk(df, cities)
    else:
        # If we have city_risk_df but no coordinates, add them from the cities list
        if 'lat' not in city_risk_df.columns or 'lon' not in city_risk_df.columns:
            # Add coordinate data to city_risk_df
            city_coords = {city['name']: (city['lat'], city['lon']) for city in cities}
            city_risk_df['lat'] = city_risk_df['city'].map(lambda x: city_coords.get(x, (None, None))[0])
            city_risk_df['lon'] = city_risk_df['city'].map(lambda x: city_coords.get(x, (None, None))[1])
    
    if city_risk_df is not None:
        # Display city risk data
        st.dataframe(city_risk_df.sort_values("weighted_risk", ascending=False))
        
        # Create city risk visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(city_risk_df["city"][:10], city_risk_df["weighted_risk"][:10])
        
        # Color bars by risk category
        for i, bar in enumerate(bars):
            if city_risk_df.iloc[i]["risk_category"] == "High":
                bar.set_color("red")
            elif city_risk_df.iloc[i]["risk_category"] == "Medium":
                bar.set_color("orange")
            else:
                bar.set_color("green")
        
        ax.set_xticklabels(city_risk_df["city"][:10], rotation=45, ha='right')
        ax.set_xlabel("City")
        ax.set_ylabel("Weighted Risk Score")
        ax.set_title("Top 10 Cities by Earthquake Risk")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display pre-rendered city risk map
        st.write("### City Risk Map")
        
        city_risk_map_path = "maps/enhanced_city_earthquake_risk_map.html"
        if os.path.exists(city_risk_map_path):
            st.components.v1.html(open(city_risk_map_path, 'r', encoding='utf-8').read(), height=600)
        else:
            st.error(f"Pre-rendered city risk map not found at {city_risk_map_path}.")

def calculate_city_risk(df, cities, radius_km=50):
    """
    Calculate earthquake risk for a list of cities
    
    Parameters:
    - df: DataFrame with earthquake data
    - cities: List of dictionaries with city information
    - radius_km: Radius around city to consider for risk calculation
    
    Returns:
    - DataFrame with city risk data
    """
    import math
    
    # Haversine formula to calculate distance
    def haversine(lon1, lat1, lon2, lat2):
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of earth in km
        return c * r
    
    city_risks = []
    
    for city in cities:
        # Find earthquakes within the radius
        nearby_earthquakes = []
        for idx, quake in df.iterrows():
            distance = haversine(city["lon"], city["lat"], quake["Longitude"], quake["Latitude"])
            if distance <= radius_km:
                quake_info = {
                    "distance": distance,
                    "magnitude": quake["Magnitude"],
                    "risk_score": quake.get("Risk_Score", 0)
                }
                # Add fault data if available
                if 'distance_to_fault' in df.columns:
                    quake_info['distance_to_fault'] = quake['distance_to_fault']
                    quake_info['nearest_fault_importance'] = quake.get('nearest_fault_importance', 1)
                nearby_earthquakes.append(quake_info)
        
        # If no earthquakes found, return minimal risk
        if not nearby_earthquakes:
            city_risks.append({
                "city": city["name"],
                "population": city["population"],
                "lat": city["lat"],
                "lon": city["lon"],
                "earthquake_count": 0,
                "avg_magnitude": 0,
                "max_magnitude": 0,
                "avg_risk_score": 0,
                "weighted_risk": 0,
                "risk_category": "Low"
            })
            continue
        
        # Calculate risk metrics
        earthquake_count = len(nearby_earthquakes)
        avg_magnitude = sum(q["magnitude"] for q in nearby_earthquakes) / earthquake_count
        max_magnitude = max(q["magnitude"] for q in nearby_earthquakes)
        
        # Calculate average risk score
        avg_risk_score = sum(q["risk_score"] for q in nearby_earthquakes) / earthquake_count
        
        # Weight by magnitude and distance
        weighted_risks = []
        for quake in nearby_earthquakes:
            # Higher magnitude and closer distance = higher risk
            weight = quake["magnitude"] * (1 / (quake["distance"] + 1))
            
            # Add fault-based risk if available
            if 'distance_to_fault' in quake:
                fault_distance_factor = 1 / (quake['distance_to_fault'] + 1)
                fault_importance_factor = quake.get('nearest_fault_importance', 1) / 5
                weight *= (1 + fault_distance_factor * fault_importance_factor)
                
            weighted_risks.append(weight)
        
        weighted_risk = sum(weighted_risks) * (city["population"] / 1000000)  # Scale by population in millions
        
        # Determine risk category
        risk_category = "Low"
        if weighted_risk > 50:
            risk_category = "High"
        elif weighted_risk > 20:
            risk_category = "Medium"
        
        city_risks.append({
            "city": city["name"],
            "population": city["population"],
            "lat": city["lat"],
            "lon": city["lon"],
            "earthquake_count": earthquake_count,
            "avg_magnitude": avg_magnitude,
            "max_magnitude": max_magnitude,
            "avg_risk_score": avg_risk_score,
            "weighted_risk": weighted_risk,
            "risk_category": risk_category
        })
    
    return pd.DataFrame(city_risks)
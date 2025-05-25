# Path: /components/model_predictor.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import datetime
from streamlit_folium import st_folium
from shapely.geometry import Point

def get_all_expected_features():
    """Get complete feature list matching trained model exactly"""
    return [
        # Base features
        'Longitude', 'Latitude', 'Depth', 'Year', 'Month', 'Day', 'DayOfWeek',
        
        # Temporal features
        'DayOfYear', 'WeekOfYear', 'IsWeekend', 'MonthSin', 'MonthCos', 
        'DayOfYearSin', 'DayOfYearCos', 'day_of_year_normalized',
        
        # Spatial features
        'PrevQuakesInGrid', 'DistFromPrev', 'DaysSinceLastQuake', 'PrevMagnitude', 
        'DepthByLat', 'DepthByLon',
        
        # Fault features
        'distance_to_fault', 'nearest_fault_importance', 'weighted_fault_risk',
        'fault_complexity_score', 'fault_count_100km', 'avg_fault_distance',
        'dominant_fault_importance', 'fault_length_sum', 'fault_intersection_count',
        'fault_intersection_importance', 'complexity_score', 'fault_importance_sum',
        
        # Density features
        'density_10km', 'density_25km', 'density_50km', 'density_100km',
        'avg_mag_10km', 'avg_mag_25km', 'avg_mag_50km', 'avg_mag_100km',
        'max_mag_10km', 'max_mag_25km', 'max_mag_50km', 'max_mag_100km',
        'mag_std_10km', 'mag_std_25km', 'mag_std_50km', 'mag_std_100km',
        'recent_activity_10km', 'recent_activity_25km', 'recent_activity_50km', 'recent_activity_100km',
        
        # Regional activity
        'regional_activity_30d', 'regional_activity_90d', 'regional_activity_365d',
        'regional_max_mag_30d', 'regional_max_mag_90d', 'regional_max_mag_365d',
        
        # Interaction features
        'depth_magnitude_interaction', 'location_depth_interaction',
        'seasonal_location_lat', 'seasonal_location_lon',
        'multi_fault_complexity', 'magnitude_fault_interaction', 'weighted_magnitude_interaction',
        
        # Polynomial features
        'poly_Depth weighted_fault_risk', 'poly_Depth fault_complexity_score',
        'poly_weighted_fault_risk fault_complexity_score'
    ]

def create_complete_feature_set(lat, lon, depth, fault_distance, fault_importance):
    """Create complete feature set with all expected features"""
    today = datetime.datetime.now()
    
    features = {}
    
    # Base features
    features.update({
        'Longitude': lon, 'Latitude': lat, 'Depth': depth,
        'Year': today.year, 'Month': today.month, 'Day': today.day,
        'DayOfWeek': today.weekday(), 'DayOfYear': today.timetuple().tm_yday,
        'WeekOfYear': today.isocalendar()[1], 'IsWeekend': 1 if today.weekday() >= 5 else 0
    })
    
    # Cyclical encoding
    features.update({
        'MonthSin': np.sin(2 * np.pi * today.month / 12),
        'MonthCos': np.cos(2 * np.pi * today.month / 12),
        'DayOfYearSin': np.sin(2 * np.pi * today.timetuple().tm_yday / 365),
        'DayOfYearCos': np.cos(2 * np.pi * today.timetuple().tm_yday / 365),
        'day_of_year_normalized': today.timetuple().tm_yday / 365.25
    })
    
    # Spatial features
    features.update({
        'PrevQuakesInGrid': 5, 'DistFromPrev': 15.0, 'DaysSinceLastQuake': 30.0,
        'PrevMagnitude': 4.5, 'DepthByLat': depth * lat, 'DepthByLon': depth * lon
    })
    
    # Enhanced fault features
    weighted_risk = fault_importance / (fault_distance + 1)
    complexity = fault_importance * np.log(fault_distance + 1)
    
    features.update({
        'distance_to_fault': fault_distance, 'nearest_fault_importance': fault_importance,
        'weighted_fault_risk': weighted_risk, 'fault_complexity_score': complexity,
        'fault_count_100km': max(1, int(5 - fault_distance/20)), 'avg_fault_distance': fault_distance,
        'dominant_fault_importance': fault_importance, 'fault_length_sum': fault_distance * 2,
        'fault_intersection_count': max(1, int(3 - fault_distance/30)),
        'fault_intersection_importance': fault_importance * 0.8,
        'complexity_score': fault_importance * fault_distance / 10,
        'fault_importance_sum': fault_importance * 1.2
    })
    
    # Density features
    density_base = 0.001
    features.update({
        'density_10km': density_base, 'density_25km': density_base * 2, 
        'density_50km': density_base * 3, 'density_100km': density_base * 4,
        'avg_mag_10km': 4.2, 'avg_mag_25km': 4.3, 'avg_mag_50km': 4.4, 'avg_mag_100km': 4.5,
        'max_mag_10km': 5.0, 'max_mag_25km': 5.2, 'max_mag_50km': 5.5, 'max_mag_100km': 6.0,
        'mag_std_10km': 0.3, 'mag_std_25km': 0.4, 'mag_std_50km': 0.5, 'mag_std_100km': 0.6,
        'recent_activity_10km': 2, 'recent_activity_25km': 4, 'recent_activity_50km': 8, 'recent_activity_100km': 15
    })
    
    # Regional activity
    features.update({
        'regional_activity_30d': 3, 'regional_activity_90d': 8, 'regional_activity_365d': 25,
        'regional_max_mag_30d': 4.5, 'regional_max_mag_90d': 5.0, 'regional_max_mag_365d': 5.5
    })
    
    # Interaction features
    features.update({
        'depth_magnitude_interaction': depth * 4.5,
        'location_depth_interaction': lat * lon * depth,
        'seasonal_location_lat': features['MonthSin'] * lat,
        'seasonal_location_lon': features['MonthCos'] * lon,
        'multi_fault_complexity': weighted_risk * features['fault_intersection_count'],
        'magnitude_fault_interaction': 4.5 / (fault_distance + 1),
        'weighted_magnitude_interaction': 4.5 * weighted_risk
    })
    
    # Polynomial features
    features.update({
        'poly_Depth weighted_fault_risk': depth * weighted_risk,
        'poly_Depth fault_complexity_score': depth * complexity,
        'poly_weighted_fault_risk fault_complexity_score': weighted_risk * complexity
    })
    
    return features

def show_model_predictor(df, pipeline, scaler=None):
    """Magnitude prediction interface"""
    st.title("Earthquake Magnitude Predictor")
    
    if pipeline is None:
        st.error("Model not loaded. Please make sure the model file exists.")
        return
    
    # Session state for tracking location
    if 'latitude' not in st.session_state:
        st.session_state.latitude = 38.5  # Default: center of Turkey
    if 'longitude' not in st.session_state:
        st.session_state.longitude = 35.5
    if 'distance_to_fault' not in st.session_state:
        st.session_state.distance_to_fault = 10.0
    if 'nearest_fault_importance' not in st.session_state:
        st.session_state.nearest_fault_importance = 3
    if 'nearest_fault_name' not in st.session_state:
        st.session_state.nearest_fault_name = "Unknown"
    
    # Import fault data
    try:
        import geopandas as gpd
        fault_gdf = gpd.read_file('data/tr_faults_imp.geojson')
        has_fault_data = True
    except Exception as e:
        fault_gdf = None
        has_fault_data = False
    
    # Create two columns for layout
    map_col, input_col = st.columns([2, 1])
    
    with map_col:
        st.subheader("Select Earthquake Location")
        
        # Create folium map
        m = folium.Map(location=[st.session_state.latitude, st.session_state.longitude], 
                      zoom_start=6)
        
        # Add current marker with a popup
        folium.Marker(
            [st.session_state.latitude, st.session_state.longitude],
            popup=f"Lat: {st.session_state.latitude:.4f}, Lon: {st.session_state.longitude:.4f}"
        ).add_to(m)
        
        # Add fault lines to map if available
        if has_fault_data:
            for _, fault in fault_gdf.iterrows():
                # Color based on importance
                importance = fault['importance']
                color = '#FF0000' if importance >= 4 else '#FFA500' if importance >= 3 else '#FFFF00'
                
                # Add the fault line
                folium.GeoJson(
                    fault.geometry,
                    name=fault['FAULT_NAME'],
                    style_function=lambda x, color=color, weight=importance*0.5: {
                        'color': color,
                        'weight': weight,
                        'opacity': 0.7
                    },
                    tooltip=f"Fault: {fault['FAULT_NAME']}, Importance: {fault['importance']}"
                ).add_to(m)
        
        # Add click event handler with st_folium
        map_data = st_folium(m, width=600, height=400)
        
        # Update lat/long if user clicks on map
        if map_data["last_clicked"]:
            # Update session state with the clicked coordinates
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]
            
            # Update session state
            st.session_state.latitude = clicked_lat
            st.session_state.longitude = clicked_lon
            
            # Calculate distance to nearest fault and importance if fault data available
            if has_fault_data:
                point = Point(clicked_lon, clicked_lat)
                
                # Calculate distance to each fault line
                distances = []
                for idx, fault in fault_gdf.iterrows():
                    fault_geom = fault.geometry
                    dist = point.distance(fault_geom)
                    distances.append((dist, idx))
                
                # Find the closest fault
                closest_dist, closest_idx = min(distances, key=lambda x: x[0])
                
                # Convert distance to kilometers (approximation)
                # 1 degree ≈ 111 km at the equator
                dist_km = closest_dist * 111
                
                # Get fault properties
                closest_fault = fault_gdf.iloc[closest_idx]
                
                # Update session state
                st.session_state.distance_to_fault = dist_km
                st.session_state.nearest_fault_importance = closest_fault.get('importance', 3)
                st.session_state.nearest_fault_name = closest_fault.get('FAULT_NAME', "Unknown")
                
                # Force refresh
                st.rerun()
        
        # Display current selected location with fault info
        if has_fault_data:
            st.info(f"📍 Selected location: {st.session_state.latitude:.4f}, {st.session_state.longitude:.4f}\n\n" +
                   f"Nearest fault: {st.session_state.nearest_fault_name}\n" +
                   f"Distance to fault: {st.session_state.distance_to_fault:.2f} km\n" +
                   f"Fault importance: {st.session_state.nearest_fault_importance}")
        else:
            st.info(f"📍 Selected location: {st.session_state.latitude:.4f}, {st.session_state.longitude:.4f}")
    
    with input_col:
        st.subheader("Earthquake Parameters")
        
        # Allow manual adjustment of coordinates
        latitude = st.number_input("Latitude", min_value=35.0, max_value=43.0, 
                                  value=st.session_state.latitude, step=0.1,
                                  key="input_latitude")
        longitude = st.number_input("Longitude", min_value=25.0, max_value=45.0, 
                                   value=st.session_state.longitude, step=0.1,
                                   key="input_longitude")
        
        # Update session state when inputs change
        st.session_state.latitude = latitude
        st.session_state.longitude = longitude
        
        # Depth with slider for better UX
        depth = st.slider("Depth (km)", min_value=0.0, max_value=100.0, value=10.0, step=5.0)
        
        # Display fault information instead of sliders
        if has_fault_data:
            st.subheader("Fault Information")
            st.write(f"**Nearest fault:** {st.session_state.nearest_fault_name}")
            st.write(f"**Distance to fault:** {st.session_state.distance_to_fault:.2f} km")
            st.write(f"**Fault importance:** {st.session_state.nearest_fault_importance}")
            
            # Risk level display with emojis
            risk_level = "High" if st.session_state.distance_to_fault < 10 else "Medium" if st.session_state.distance_to_fault < 25 else "Low"
            risk_color = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
            st.write(f"**Risk Level:** {risk_color} {risk_level}")
    
    # Prediction button
    if st.button("🔮 Predict Magnitude", type="primary"):
        with st.spinner("🧠 Calculating prediction..."):
            # Create complete feature set
            features = create_complete_feature_set(
                latitude, longitude, depth,
                st.session_state.distance_to_fault,
                st.session_state.nearest_fault_importance
            )
            
            # Make prediction
            prediction = predict_magnitude(pipeline, features)
            
            if prediction is not None:
                # Display results
                st.success(f"🎯 **Predicted Magnitude: {prediction:.2f}**")
                
                # Enhanced gauge visualization
                fig, ax = plt.subplots(figsize=(10, 3))
                cmap = plt.cm.get_cmap('RdYlBu_r')
                ax.barh(0, 10, color='lightgray', height=0.6, alpha=0.3)
                ax.barh(0, min(prediction, 10), color=cmap(prediction/10), height=0.6)
                
                # Add magnitude markers
                for i in range(11):
                    ax.axvline(i, color='black', alpha=0.3, linewidth=0.5)
                    ax.text(i, -0.4, str(i), ha='center', fontsize=10)
                
                # Prediction pointer
                ax.plot([prediction, prediction], [-0.2, 0.8], 'k-', linewidth=3)
                ax.text(prediction, 1.0, f'{prediction:.2f}', ha='center', fontweight='bold', fontsize=12)
                
                # Styling
                ax.set_xlim(0, 10)
                ax.set_ylim(-0.5, 1.2)
                ax.axis('off')
                ax.set_title('Magnitude Prediction', fontsize=14, fontweight='bold')
                
                st.pyplot(fig)
                
                # Enhanced interpretation with emojis
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction < 5.0:
                        st.info("🟢 **Moderate Earthquake**\nMinor to light damage expected")
                    elif prediction < 6.0:
                        st.warning("🟡 **Strong Earthquake**\nModerate damage possible")
                    else:
                        st.error("🔴 **Major Earthquake**\nSerious damage expected")
                
                with col2:
                    # Risk assessment based on location
                    fault_risk = st.session_state.distance_to_fault
                    if fault_risk < 10:
                        st.error("⚠️ **High Fault Risk**\nVery close to major fault")
                    elif fault_risk < 25:
                        st.warning("⚡ **Medium Fault Risk**\nModerate distance from fault")
                    else:
                        st.success("✅ **Lower Fault Risk**\nFar from major faults")
                
                with col3:
                    # Population impact estimate
                    if prediction >= 6.0:
                        st.error("🏘️ **High Impact**\nWide area affected")
                    elif prediction >= 5.0:
                        st.warning("🏠 **Medium Impact**\nLocal area affected")
                    else:
                        st.info("🏡 **Low Impact**\nMinimal area affected")
    
    # Show feature importance in an expander
    with st.expander("Feature Importance Analysis", expanded=False):
        try:
            model_component = pipeline.named_steps['model']
            if hasattr(model_component, 'feature_importances_'):
                # Get the list of features used during training
                feature_names = get_all_expected_features()
                
                # Trim or pad feature names list to match feature importances length
                importance_len = len(model_component.feature_importances_)
                
                if len(feature_names) > importance_len:
                    feature_names = feature_names[:importance_len]
                elif len(feature_names) < importance_len:
                    feature_names = feature_names + [f"Feature_{i}" for i in range(len(feature_names), importance_len)]
                
                # Create feature importance DataFrame
                importance_data = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model_component.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_data.head(10), ax=ax)
                ax.set_title('Top 10 Feature Importances')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.warning("Feature importance visualization is not available.")

def predict_magnitude(pipeline, features):
    """Predict earthquake magnitude using the pipeline"""
    if pipeline is None:
        st.error("Model not found. Please train a model first.")
        return None
    
    try:
        # Create DataFrame with all expected features
        expected_features = get_all_expected_features()
        df = pd.DataFrame([features])
        
        # Ensure all columns exist
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Reorder columns to match training
        df = df[expected_features]
        
        # Make prediction (pipeline handles preprocessing)
        prediction = pipeline.predict(df)
        
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None
# components/model_predictor.py
# Path: /components/model_predictor.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
import datetime
from streamlit_folium import st_folium
from shapely.geometry import Point

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
    
    # Create features dictionary
    features = {
        'Longitude': longitude,
        'Latitude': latitude,
        'Depth': depth,
        'distance_to_fault': st.session_state.distance_to_fault,
        'nearest_fault_importance': st.session_state.nearest_fault_importance
    }
    
    # Add date-related data
    today = datetime.datetime.now()
    features['Year'] = today.year
    features['Month'] = today.month
    features['Day'] = today.day
    features['DayOfWeek'] = today.weekday()
    
    # Prediction button
    if st.button("Predict Magnitude", type="primary"):
        with st.spinner("Calculating prediction..."):
            # Make prediction
            prediction = predict_magnitude(pipeline, features)
            
            if prediction is not None:
                # Create result container with highlight
                result_container = st.container()
                with result_container:
                    # Display prediction with colorful gauge
                    st.subheader(f"Predicted Magnitude: {prediction:.2f}")
                    
                    # Add a colorful gauge representation
                    fig, ax = plt.subplots(figsize=(8, 2))
                    
                    # Create a gauge-like visualization
                    cmap = sns.color_palette("YlOrRd", as_cmap=True)
                    ax.barh(0, 10, color='lightgray', height=0.4)
                    ax.barh(0, min(prediction, 10), color=cmap(prediction/10), height=0.4)
                    
                    # Add magnitude scale
                    for i in range(11):
                        ax.text(i, -0.2, str(i), ha='center', fontsize=10)
                    
                    # Add pointer to the predicted value
                    ax.plot([prediction, prediction], [-0.1, 0.3], 'k-', linewidth=2)
                    
                    # Remove axes
                    ax.axis('off')
                    ax.set_xlim(0, 10)
                    ax.set_ylim(-0.3, 0.5)
                    
                    st.pyplot(fig)
                    
                    # Interpret the prediction
                    if prediction < 5.0:
                        st.info("This is predicted to be a moderate earthquake.")
                    elif prediction < 6.0:
                        st.warning("This is predicted to be a strong earthquake.")
                    else:
                        st.error("This is predicted to be a major earthquake!")
    
    # Show feature importance in an expander
    with st.expander("Feature Importance Analysis", expanded=False):
        try:
            model_component = pipeline.named_steps['model']
            if hasattr(model_component, 'feature_importances_'):
                # Get the list of features used during training
                feature_names = ['Longitude', 'Latitude', 'Depth', 'Year', 'Month', 'Day', 'DayOfWeek',
                                'DayOfYear', 'WeekOfYear', 'IsWeekend', 'MonthSin', 'MonthCos', 
                                'DayOfYearSin', 'DayOfYearCos', 'PrevQuakesInGrid', 'DistFromPrev',
                                'DaysSinceLastQuake', 'PrevMagnitude', 'DepthByLat', 'DepthByLon']
                
                # Add fault features if they exist in the dataset
                if 'distance_to_fault' in df.columns:
                    feature_names.extend(['distance_to_fault', 'nearest_fault_importance', 
                                        'fault_count_50km', 'fault_length_50km', 'fault_density',
                                        'magnitude_fault_interaction'])
                
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
    
    # Convert features to DataFrame with only provided features
    df = pd.DataFrame([features])
    
    # Get all expected feature names the pipeline was trained with
    required_features = ['Longitude', 'Latitude', 'Depth', 'Year', 'Month', 'Day', 
                        'DayOfWeek', 'DayOfYear', 'WeekOfYear', 'IsWeekend', 
                        'MonthSin', 'MonthCos', 'DayOfYearSin', 'DayOfYearCos',
                        'PrevQuakesInGrid', 'DistFromPrev', 'DaysSinceLastQuake', 
                        'PrevMagnitude', 'DepthByLat', 'DepthByLon', 
                        'fault_count_50km', 'fault_length_50km', 'fault_density',
                        'magnitude_fault_interaction']
    
    # Set default values for missing features
    for feature in required_features:
        if feature not in df.columns:
            # If temporal feature, use reasonable defaults
            if feature in ['DayOfYear', 'WeekOfYear']:
                today = datetime.datetime.now()
                df[feature] = today.timetuple().tm_yday if feature == 'DayOfYear' else today.isocalendar()[1]
            elif feature == 'IsWeekend':
                df[feature] = 1 if df['DayOfWeek'].iloc[0] >= 5 else 0
            # For cyclical features
            elif feature == 'MonthSin':
                df[feature] = np.sin(2 * np.pi * df['Month'] / 12)
            elif feature == 'MonthCos':
                df[feature] = np.cos(2 * np.pi * df['Month'] / 12)
            elif feature == 'DayOfYearSin':
                day_of_year = datetime.datetime.now().timetuple().tm_yday
                df[feature] = np.sin(2 * np.pi * day_of_year / 365)
            elif feature == 'DayOfYearCos':
                day_of_year = datetime.datetime.now().timetuple().tm_yday
                df[feature] = np.cos(2 * np.pi * day_of_year / 365)
            # For interaction features
            elif feature == 'DepthByLat':
                df[feature] = df['Depth'] * df['Latitude']
            elif feature == 'DepthByLon':
                df[feature] = df['Depth'] * df['Longitude']
            elif feature == 'magnitude_fault_interaction' and 'distance_to_fault' in df.columns:
                df[feature] = 5 / (df['distance_to_fault'] + 1)  # Assuming magnitude ~5
            # Default values for other features
            else:
                df[feature] = 0
    
    # Make prediction (pipeline handles preprocessing)
    prediction = pipeline.predict(df)
    
    return prediction[0]
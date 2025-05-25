# Path: /utils/model_utils.py
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import datetime
import os

def get_expected_features():
    """Get expected feature list for enhanced models"""
    base_features = [
        'Longitude', 'Latitude', 'Depth', 'Year', 'Month', 'Day', 'DayOfWeek'
    ]
    
    temporal_features = [
        'DayOfYear', 'WeekOfYear', 'IsWeekend', 'MonthSin', 'MonthCos', 
        'DayOfYearSin', 'DayOfYearCos', 'day_of_year_normalized'
    ]
    
    spatial_features = [
        'PrevQuakesInGrid', 'DistFromPrev', 'DaysSinceLastQuake', 
        'PrevMagnitude', 'DepthByLat', 'DepthByLon'
    ]
    
    fault_features = [
        'distance_to_fault', 'nearest_fault_importance', 'weighted_fault_risk',
        'fault_complexity_score', 'fault_count_100km', 'avg_fault_distance',
        'dominant_fault_importance', 'fault_length_sum', 'fault_intersection_count',
        'fault_intersection_importance', 'complexity_score', 'fault_importance_sum'
    ]
    
    density_features = [
        'density_10km', 'density_25km', 'density_50km', 'density_100km',
        'avg_mag_10km', 'avg_mag_25km', 'avg_mag_50km', 'avg_mag_100km',
        'max_mag_10km', 'max_mag_25km', 'max_mag_50km', 'max_mag_100km',
        'mag_std_10km', 'mag_std_25km', 'mag_std_50km', 'mag_std_100km',
        'recent_activity_10km', 'recent_activity_25km', 'recent_activity_50km', 'recent_activity_100km'
    ]
    
    temporal_advanced = [
        'regional_activity_30d', 'regional_activity_90d', 'regional_activity_365d',
        'regional_max_mag_30d', 'regional_max_mag_90d', 'regional_max_mag_365d'
    ]
    
    interaction_features = [
        'depth_magnitude_interaction', 'location_depth_interaction',
        'seasonal_location_lat', 'seasonal_location_lon',
        'multi_fault_complexity', 'magnitude_fault_interaction', 'weighted_magnitude_interaction'
    ]
    
    # Polynomial features from notebook
    polynomial_features = [
        'poly_Depth weighted_fault_risk', 'poly_Depth fault_complexity_score',
        'poly_weighted_fault_risk fault_complexity_score'
    ]
    
    # Combine all features
    all_features = (base_features + temporal_features + spatial_features + 
                   fault_features + density_features + temporal_advanced + 
                   interaction_features + polynomial_features)
    
    return all_features

def get_feature_default(feature_name, user_features):
    """Get sensible default values for missing features"""
    today = datetime.datetime.now()
    lat = user_features.get('Latitude', 38.5)
    lon = user_features.get('Longitude', 35.5)
    depth = user_features.get('Depth', 10.0)
    fault_dist = user_features.get('distance_to_fault', 10.0)
    fault_imp = user_features.get('nearest_fault_importance', 3)
    
    defaults = {
        # Temporal features
        'DayOfYear': today.timetuple().tm_yday,
        'WeekOfYear': today.isocalendar()[1],
        'IsWeekend': 1 if today.weekday() >= 5 else 0,
        'MonthSin': np.sin(2 * np.pi * today.month / 12),
        'MonthCos': np.cos(2 * np.pi * today.month / 12),
        'DayOfYearSin': np.sin(2 * np.pi * today.timetuple().tm_yday / 365),
        'DayOfYearCos': np.cos(2 * np.pi * today.timetuple().tm_yday / 365),
        'day_of_year_normalized': today.timetuple().tm_yday / 365.25,
        
        # Spatial defaults
        'PrevQuakesInGrid': 5,
        'DistFromPrev': 15.0,
        'DaysSinceLastQuake': 30.0,
        'PrevMagnitude': 4.5,
        'DepthByLat': depth * lat,
        'DepthByLon': depth * lon,
        
        # Enhanced fault features
        'weighted_fault_risk': fault_imp / (fault_dist + 1),
        'fault_complexity_score': fault_imp * np.log(fault_dist + 1),
        'fault_count_100km': max(1, int(5 - fault_dist/20)),
        'avg_fault_distance': fault_dist,
        'dominant_fault_importance': fault_imp,
        'fault_length_sum': fault_dist * 2,
        'fault_intersection_count': max(1, int(3 - fault_dist/30)),
        'fault_intersection_importance': fault_imp * 0.8,
        'complexity_score': fault_imp * fault_dist / 10,
        'fault_importance_sum': fault_imp * 1.2,
        
        # Density features (reasonable defaults)
        'density_10km': 0.001, 'density_25km': 0.002, 'density_50km': 0.003, 'density_100km': 0.004,
        'avg_mag_10km': 4.2, 'avg_mag_25km': 4.3, 'avg_mag_50km': 4.4, 'avg_mag_100km': 4.5,
        'max_mag_10km': 5.0, 'max_mag_25km': 5.2, 'max_mag_50km': 5.5, 'max_mag_100km': 6.0,
        'mag_std_10km': 0.3, 'mag_std_25km': 0.4, 'mag_std_50km': 0.5, 'mag_std_100km': 0.6,
        'recent_activity_10km': 2, 'recent_activity_25km': 4, 'recent_activity_50km': 8, 'recent_activity_100km': 15,
        
        # Regional activity
        'regional_activity_30d': 3, 'regional_activity_90d': 8, 'regional_activity_365d': 25,
        'regional_max_mag_30d': 4.5, 'regional_max_mag_90d': 5.0, 'regional_max_mag_365d': 5.5,
        
        # Interaction features
        'depth_magnitude_interaction': depth * 4.5,  # Assume magnitude ~4.5
        'location_depth_interaction': lat * lon * depth,
        'seasonal_location_lat': np.sin(2 * np.pi * today.month / 12) * lat,
        'seasonal_location_lon': np.cos(2 * np.pi * today.month / 12) * lon,
        'multi_fault_complexity': (fault_imp / (fault_dist + 1)) * max(1, int(3 - fault_dist/30)),
        'magnitude_fault_interaction': 4.5 / (fault_dist + 1),
        'weighted_magnitude_interaction': 4.5 * fault_imp / (fault_dist + 1),
        
        # Polynomial features
        'poly_Depth weighted_fault_risk': depth * (fault_imp / (fault_dist + 1)),
        'poly_Depth fault_complexity_score': depth * (fault_imp * np.log(fault_dist + 1)),
        'poly_weighted_fault_risk fault_complexity_score': (fault_imp / (fault_dist + 1)) * (fault_imp * np.log(fault_dist + 1))
    }
    
    return defaults.get(feature_name, 0)

def predict_magnitude(model, scaler, user_features):
    """Predict earthquake magnitude with flexible feature handling"""
    if model is None:
        st.error("Model not loaded.")
        return None
    
    try:
        # Convert to DataFrame for pipeline compatibility
        df = pd.DataFrame([user_features])
        
        # Get expected features from the model
        expected_features = get_expected_features()
        
        # Add missing features with defaults
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = get_feature_default(feature, user_features)
        
        # Ensure column order matches training
        df = df[expected_features]
        
        # Make prediction (pipeline handles preprocessing)
        prediction = model.predict(df)
        return prediction[0]
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def get_feature_importance(model):
    """Get feature importance with flexible handling"""
    if model is None:
        return None
    
    try:
        # Handle pipeline models
        if hasattr(model, 'named_steps'):
            model_component = model.named_steps.get('model')
        else:
            model_component = model
        
        if not hasattr(model_component, 'feature_importances_'):
            return None
        
        # Get feature importances
        importances = model_component.feature_importances_
        expected_features = get_expected_features()
        
        # Match feature names to importances
        feature_names = expected_features[:len(importances)]
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return feature_importance
        
    except Exception as e:
        st.warning(f"Feature importance not available: {str(e)}")
        return None

def validate_user_inputs(latitude, longitude, depth):
    """Validate user input parameters"""
    errors = []
    
    if not (35.0 <= latitude <= 43.0):
        errors.append("Latitude must be between 35.0 and 43.0 (Turkey bounds)")
    
    if not (25.0 <= longitude <= 45.0):
        errors.append("Longitude must be between 25.0 and 45.0 (Turkey bounds)")
    
    if not (0.0 <= depth <= 100.0):
        errors.append("Depth must be between 0.0 and 100.0 km")
    
    return errors
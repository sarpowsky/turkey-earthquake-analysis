# utils/model_utils.py
# Path: /utils/model_utils.py
import pandas as pd
import numpy as np
import joblib
import streamlit as st

def predict_magnitude(model, scaler, user_features):
    """Predict earthquake magnitude using only the model"""
    if model is None:
        st.error("Model not loaded.")
        return None
    
    try:
        # Create array with all 26 features the model expects with reasonable defaults
        features_array = np.zeros(26)
        
        # Base features (0-2)
        features_array[0] = user_features.get('Longitude', 35.0)
        features_array[1] = user_features.get('Latitude', 38.5)
        features_array[2] = user_features.get('Depth', 10.0)
        
        # Time features (3-6)
        features_array[3] = user_features.get('Year', 2023)
        features_array[4] = user_features.get('Month', 6)
        features_array[5] = user_features.get('Day', 15)
        features_array[6] = user_features.get('DayOfWeek', 3)
        
        # Previous earthquake features (7-10)
        features_array[7] = user_features.get('PrevQuakesInGrid', 5)
        features_array[8] = user_features.get('DistFromPrev', 15.0)
        features_array[9] = user_features.get('DaysSinceLastQuake', 30.0)
        features_array[10] = user_features.get('PrevMagnitude', 4.5)
        
        # Interaction features (11-12)
        features_array[11] = user_features.get('DepthByLat', features_array[2] * features_array[1])
        features_array[12] = user_features.get('DepthByLon', features_array[2] * features_array[0])
        
        # Fault-related features (13-17)
        features_array[13] = user_features.get('distance_to_fault', 10.0)
        features_array[14] = user_features.get('nearest_fault_importance', 3)
        features_array[15] = user_features.get('fault_count_50km', 2)
        features_array[16] = user_features.get('fault_length_50km', 40.0)
        features_array[17] = user_features.get('fault_density', 0.015)
        
        # Cyclical features (18-21)
        features_array[18] = user_features.get('MonthSin', np.sin(2 * np.pi * features_array[4]/12))
        features_array[19] = user_features.get('MonthCos', np.cos(2 * np.pi * features_array[4]/12))
        features_array[20] = user_features.get('DayOfYearSin', np.sin(2 * np.pi * 166/365))
        features_array[21] = user_features.get('DayOfYearCos', np.cos(2 * np.pi * 166/365))
        
        # Risk features (22-24)
        features_array[22] = user_features.get('Fault_Distance_Score', 0.35)
        features_array[23] = user_features.get('Fault_Importance_Score', 0.48)
        features_array[24] = user_features.get('Risk_Score', 5.0)
        
        # Interaction features (25)
        features_array[25] = user_features.get('magnitude_fault_interaction', 0.45)
        
        # Convert to proper shape for prediction
        input_data = features_array.reshape(1, -1)
        
        # Make prediction directly
        prediction = model.predict(input_data)
        return prediction[0]
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def get_feature_importance(model):
    """Get feature importance from the model if available"""
    if model is None or not hasattr(model, 'feature_importances_'):
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create more meaningful feature names
    feature_names = [
        'Longitude', 'Latitude', 'Depth', 'Year', 'Month', 
        'Day', 'DayOfWeek', 'PrevQuakesInGrid', 'DistFromPrev', 
        'DaysSinceLastQuake', 'PrevMagnitude', 'DepthByLat', 
        'DepthByLon', 'distance_to_fault', 'nearest_fault_importance',
        'fault_count_50km', 'fault_length_50km', 'fault_density',
        'MonthSin', 'MonthCos', 'DayOfYearSin', 'DayOfYearCos',
        'Fault_Distance_Score', 'Fault_Importance_Score', 'Risk_Score',
        'magnitude_fault_interaction'
    ]
    
    # Make sure we have the right number of feature names
    if len(feature_names) != len(importances):
        feature_names = [f'Feature {i+1}' for i in range(len(importances))]
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names[:len(importances)],
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return feature_importance
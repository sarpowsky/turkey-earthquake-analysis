# utils/data_loader.py
# Path: /utils/data_loader.py
import pandas as pd
import numpy as np
import joblib
import os

def load_earthquake_data():
    """Load the processed earthquake dataset"""
    try:
        # Try to load the clustered data first
        return pd.read_csv('produced_data/earthquake_clusters.csv')
    except FileNotFoundError:
        # Fallback to clean data
        try:
            return pd.read_csv('produced_data/clean_earthquake_data.csv')
        except FileNotFoundError:
            # Original data as last resort
            return pd.read_csv('data/earthquake_data.csv')

def load_fault_data():
    """Load the fault line data"""
    import geopandas as gpd
    try:
        return gpd.read_file('data/tr_faults_imp.geojson')
    except:
        return None

def load_model():
    """Load the trained earthquake prediction model"""
    try:
        model = joblib.load('models/earthquake_magnitude_model.pkl')
        scaler = joblib.load('models/earthquake_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None
        
def get_city_risk_data():
    """Load or generate city risk assessment data"""
    try:
        return pd.read_csv('produced_data/city_risk_assessment.csv')
    except FileNotFoundError:
        return None
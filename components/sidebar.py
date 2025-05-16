# components/sidebar.py
# Path: /components/sidebar.py
import streamlit as st

def create_sidebar():
    """Create the navigation sidebar"""
    st.sidebar.title("Earthquake Analysis")
    
    # Navigation options
    pages = {
        "Home": "👋",
        "Data Explorer": "📊",
        "Map Viewer": "🗺️",
        "Magnitude Predictor": "📈",
        "Cluster Analysis": "🔍",
        "Risk Assessment": "⚠️"
    }
    
    # Selection
    selection = st.sidebar.radio("Go to", list(pages.keys()), format_func=lambda x: f"{pages[x]} {x}")
    
    # Display app info
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app analyzes Turkey's earthquake data using both supervised and unsupervised "
        "machine learning techniques to predict earthquake magnitudes and identify risk zones."
    )
    
    return selection
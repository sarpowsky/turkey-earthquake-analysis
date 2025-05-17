# components/sidebar.py
# Path: /components/sidebar.py
import streamlit as st

def create_sidebar():
    """Create the navigation sidebar"""
    # Apply custom sidebar styling
    st.markdown("""
    <style>
    .sidebar-title {
        color: #E30A17;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
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
        "Turquake analyzes Turkey's earthquake data using both supervised and unsupervised "
        "machine learning techniques to predict earthquake magnitudes and identify risk zones."
    )
    
    return selection
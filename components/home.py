# components/home.py
# Path: /components/home.py
import streamlit as st

def show_home():
    """Home page with project overview"""
    st.title("Turkish Earthquake Analysis Project")
    
    st.write("""
    This application presents a comprehensive analysis of Turkish earthquakes using machine learning techniques.
    The project uses both supervised and unsupervised learning to predict earthquake magnitudes and identify risk zones.
    """)
    
    # Project overview
    st.subheader("Project Overview")
    st.write("""
    - **Data Source**: AFAD earthquake dataset of Turkish earthquakes (>4.0 magnitude) from the last 100 years
    - **Primary Goal**: Predict earthquake magnitude using supervised learning (regression)
    - **Secondary Goal**: Identify fault line risk zones using clustering methods
    """)
    
    # Features overview
    st.subheader("Application Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Data Exploration")
        st.write("""
        - Basic dataset statistics and visualizations
        - Temporal and geographic analysis of earthquakes
        - Magnitude and depth relationships
        """)
        
        st.markdown("#### Interactive Maps")
        st.write("""
        - Earthquake distribution maps
        - Cluster visualization
        - Risk zone identification
        """)
    
    with col2:
        st.markdown("#### Magnitude Prediction")
        st.write("""
        - Machine learning model for magnitude prediction
        - Feature importance analysis
        - Interactive prediction interface
        """)
        
        st.markdown("#### Risk Assessment")
        st.write("""
        - Risk zone classification
        - City-specific risk analysis
        - Population impact assessment
        """)
    
    # Navigation instructions
    st.info("Use the sidebar to navigate between different sections of the application.")
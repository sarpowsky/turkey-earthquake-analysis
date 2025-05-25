# components/home.py
# Path: /components/home.py
import streamlit as st
import os

def show_home():
    """Home page with project overview"""
    # Apply custom CSS for better visual design
    apply_custom_css()
    
    # Main title and introduction
    st.markdown("<div class='title-container'><h1 class='main-title'><span class='turq'>Turq</span><span class='uake'>uake</span></h1></div>", unsafe_allow_html=True)
    
    st.write("""
    This application presents a comprehensive analysis of Turkey's earthquakes using machine learning techniques.
    The project combines supervised learning for magnitude prediction and unsupervised learning to identify risk zones.
    """)

    st.markdown("""
    <div class='personal-note'>
    <i>As someone passionate about ML, I created Turquake to help 
    visualize earthquake patterns in Turkey and predict future magnitudes using both supervised 
    and unsupervised learning techniques.</i>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(""" """, unsafe_allow_html=True)
    
    # Show map preview if available
    display_map_preview()
    
    # Project overview section
    st.markdown("<h2 class='section-header'>Project Overview</h2>", unsafe_allow_html=True)
    
    # Display key metrics in three columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Data Timespan", value="100+ Years")
    with col2:
        st.metric(label="Min. Magnitude", value="4.0+")
    with col3:
        st.metric(label="ML Models", value="Multiple")
    
    # Project goals
    st.markdown("""
    - **📊 Data Source**: AFAD earthquake dataset of Turkish earthquakes (>4.0 magnitude) from the last 100 years
    - **🎯 Primary Goal**: Predict earthquake magnitude using supervised learning (regression)
    - **🔍 Secondary Goal**: Identify fault line risk zones using clustering methods
    """)
    
    # Features overview section
    st.markdown("<h2 class='section-header'>Application Features</h2>", unsafe_allow_html=True)
    
    # Display features in a 2-column layout
    display_features_grid()
    
    # Navigation instructions
    st.markdown("<div class='nav-info'><i>🧭 **Navigation:** Use the sidebar to navigate between different sections of the application.</i></div>", unsafe_allow_html=True)

def apply_custom_css():
    """Apply custom CSS styling for better visual appearance"""
    st.markdown("""
    <style>
    .title-container {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .main-title {
        font-size: 3rem;
        margin-bottom: 1.5rem;
    }
    .turq {
        color: #E30A17; /* Turkish flag red */
        font-weight: bold;
    }
    .uake {
        color: #FFFFFF; /* White */
        font-weight: bold;
    }
    .section-header {
        color: #E30A17; /* Turkish flag red */
        border-bottom: 2px solid #E30A17;
        padding-bottom: 0.5rem;
    }
    .feature-title {
        color: #FFFFFF; /* White */
        font-weight: bold;
    }
    .nav-info {
        background-color: rgba(227, 10, 23, 0.1); /* Light red background */
        border-left: 4px solid #E30A17;
        padding: 15px;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .personal-note {
        background-color: rgba(255, 255, 255, 0.1); 
        border-left: 4px solid #E30A17;
        padding: 15px;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def display_map_preview():
    """Display a map preview if available or show placeholder"""
    # Check if map file exists
    if os.path.exists("maps/enhanced_earthquake_map.html"):
        st.components.v1.html(open("maps/enhanced_earthquake_map.html", 'r', encoding='utf-8').read(), height=300)
    elif os.path.exists("maps/enhanced_earthquake_risk_map.html"):
        st.components.v1.html(open("maps/enhanced_earthquake_risk_map.html", 'r', encoding='utf-8').read(), height=300)

def display_features_grid():
    """Display application features in a responsive grid layout"""
    # Create two columns for feature display
    col1, col2 = st.columns(2)
    
    with col1:
        # Data Exploration
        st.markdown("<h4 class='feature-title'>📊 Data Exploration</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Dataset statistics and visualizations
        - Temporal and geographic analysis
        - Magnitude and depth relationships
        - Correlation analysis
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Interactive Maps
        st.markdown("<h4 class='feature-title'>🗺️ Interactive Maps</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Earthquake distribution maps
        - Cluster visualization
        - Risk zone identification
        - Fault line overlays
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Magnitude Prediction
        st.markdown("<h4 class='feature-title'>📈 Magnitude Prediction</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Machine learning model for prediction
        - Feature importance analysis
        - Interactive prediction interface
        - Model performance metrics
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Risk Assessment
        st.markdown("<h4 class='feature-title'>⚠️ Risk Assessment</h4>", unsafe_allow_html=True)
        st.markdown("""
        - Risk zone classification
        - City-specific risk analysis
        - Population impact assessment
        - Historical pattern recognition
        """)
        st.markdown("</div>", unsafe_allow_html=True)
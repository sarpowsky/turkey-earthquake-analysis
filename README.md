# Turquake: Turkish Earthquake Analysis & Prediction

## Project Overview

Turquake is a comprehensive machine learning project focused on analyzing earthquake patterns in Turkey and predicting earthquake magnitudes. The project combines supervised learning techniques for magnitude prediction with unsupervised learning methods for identifying seismic risk zones and regional patterns.

### Key Features
- **Interactive Data Exploration**: Visualize earthquake distribution across time and space
- **Magnitude Prediction**: Machine learning model to predict earthquake magnitudes based on location, depth, and fault proximity
- **Risk Zone Identification**: Clustering analysis to identify high-risk seismic regions
- **City-Specific Risk Assessment**: Population-weighted analysis of earthquake risk for major Turkish cities
- **Interactive Maps**: Geographic visualization of earthquake patterns and fault systems
- **Web Application**: User-friendly Streamlit interface for exploring all analyses

## Technical Implementation

### Dataset
The primary dataset used is the AFAD (Disaster and Emergency Management Presidency) Turkish earthquake catalog containing events with magnitude >4.0 from approximately the last 100 years. This dataset includes:

- Geographic coordinates (longitude, latitude)
- Earthquake magnitude and depth
- Date and time information
- Location descriptions
- Fault system data with importance classifications

### Methodology

#### 1. Supervised Learning (Earthquake Magnitude Prediction)
I developed a regression model to predict earthquake magnitudes based on:
- Geographic location
- Depth measurements
- Fault proximity and importance
- Temporal patterns and seasonality
- Historical seismic activity in the region

The supervised approach involved:
- Comprehensive feature engineering including cyclical encoding of temporal variables
- Testing multiple regression algorithms (Linear, Ridge, Random Forest, Gradient Boosting)
- Hyperparameter optimization using RandomizedSearchCV
- Model evaluation with RMSE, MAE, and R² metrics
- Feature importance analysis to identify key predictive factors

#### 2. Unsupervised Learning (Seismic Pattern Analysis)
I implemented clustering algorithms to identify natural earthquake patterns and risk zones:
- K-Means clustering to identify distinct earthquake zones
- DBSCAN to detect dense clusters and outlier events
- PCA for dimensionality reduction and visualization
- Risk scoring based on magnitude, density, and fault proximity

#### 3. Web Application Development
The project is deployed as an interactive Streamlit application with:
- Multi-page navigation
- Interactive maps using Folium
- Dynamic filtering and visualization options
- Predictive interface for magnitude estimation
- Risk assessment visualization for major Turkish cities

### Technical Stack
- **Python**: Core programming language
- **Data Processing**: Pandas, NumPy, GeoPandas
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Deep Learning**: PyTorch (GPU acceleration)
- **Visualization**: Matplotlib, Seaborn, Plotly, Folium
- **Web Application**: Streamlit
- **Version Control**: Git

## Results & Findings

### Supervised Learning
- The best-performing model for magnitude prediction was XGBoost with RMSE of 0.2309
- Most important features for prediction included: Year, distance_to_fault,magnitude_fault_interaction, Depth, DepthByLat, nearest_fault_importance 
- The model showed [Your Observation about Performance with Different Magnitude Ranges]

### Unsupervised Learning
- K-Means clustering identified High-171 Medium-6501,Low-3157 distinct earthquake risk zones across Turkey
- Clear correlation between earthquake clusters and major fault systems
- High-risk zones primarily concentrated along the North Anatolian Fault along with high population cities
- Cities with highest earthquake risk: Istanbul, Izmir, Van, Sakarya, Bursa

## Future Work & Improvements
- Integration of additional geophysical data sources
- Implementation of sequence-based models (LSTM/RNN) for temporal pattern analysis
- Development of real-time prediction capabilities
- Extension of analysis to include earthquake damage estimation

## Project Structure
```
turquake/
├── app.py                 # Main Streamlit application
├── components/            # UI components
│   ├── clustering.py      # Cluster analysis page
│   ├── data_explorer.py   # Data exploration page
│   ├── home.py            # Home page
│   ├── map_viewer.py      # Map visualization page
│   ├── model_predictor.py # Magnitude prediction interface
│   ├── risk_assessment.py # Risk assessment page
│   └── sidebar.py         # Navigation sidebar
├── utils/                 # Utility functions
│   ├── data_loader.py     # Data loading utilities
│   ├── map_utils.py       # Map generation utilities
│   └── model_utils.py     # ML model utilities
├── data/                  # Data files (to be added)
│   ├── earthquake_data.csv # Main earthquake dataset 
│   └── tr_faults_imp.geojson # Turkey fault lines data
├── models/                # Saved models
├── maps/                  # Pre-rendered maps
├── produced_data/         # Processed datasets
├── supervised.ipynb       # Magnitude prediction notebook
└── unsupervised.ipynb     # Clustering analysis notebook
```

### Kaggle Notebook Links
* Supervised Learning Notebook: https://www.kaggle.com/code/sarpowsky/supervised
* Unsupervised Learning Notebook: https://www.kaggle.com/code/sarpowsky/unsupervised

### Bonus Implementations
This project includes all three bonus implementation requirements:
1. **Unsupervised Learning**: Implemented K-Means and DBSCAN clustering on the same earthquake dataset
2. **GPU Acceleration**: Added PyTorch-based neural network with CUDA support for magnitude prediction
3. **Web Application Deployment**: Created an interactive Streamlit application for exploring all analyses

### About Me
I created this project as part of the Global AI Hub Machine Learning Bootcamp. As a second-year computer engineering student passionate about ML and deep learning, I designed this application to help visualize earthquake patterns in Turkey and predict future magnitudes using both supervised and unsupervised learning techniques.

### Acknowledgments
* AFAD (Disaster and Emergency Management Presidency) for the earthquake catalog data
* Global AI Hub Machine Learning Bootcamp for the project opportunity
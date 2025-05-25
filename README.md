# Turquake: Turkish Earthquake Analysis & Prediction

## Project Overview

Turquake is a comprehensive machine learning project focused on analyzing earthquake patterns in Turkey and predicting earthquake magnitudes. The project combines supervised learning techniques for magnitude prediction with unsupervised learning methods for identifying seismic risk zones and regional patterns.

### Key Features
- **Interactive Data Exploration**: Visualize earthquake distribution across time and space
- **Magnitude Prediction**: Machine learning model to predict earthquake magnitudes based on location, depth, and comprehensive fault analysis
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
- Geographic location and depth measurements
- Multi-fault weighted risk analysis incorporating distance and importance of multiple nearby faults
- Spatial density patterns at multiple scales (10km, 25km, 50km, 100km radii)
- Temporal patterns including cyclical encoding and regional activity windows
- Fault complexity scores measuring geological interaction patterns

The supervised approach involved:
- Comprehensive feature engineering creating 80+ geological and temporal features
- Testing multiple regression algorithms (Linear, Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM)
- Hyperparameter optimization using RandomizedSearchCV with 50+ iterations
- Model evaluation with RMSE, MAE, and R² metrics
- Feature importance analysis identifying key predictive factors

#### 2. Unsupervised Learning (Seismic Pattern Analysis)
I implemented clustering algorithms to identify natural earthquake patterns and risk zones:
- K-Means clustering to identify distinct earthquake zones using geological features
- DBSCAN to detect dense clusters and outlier events
- PCA for dimensionality reduction and pattern visualization
- Risk scoring based on magnitude, density, fault proximity, and geological complexity

#### 3. Web Application Development
The project is deployed as an interactive Streamlit application with:
- Multi-page navigation system
- Interactive maps using Folium with comprehensive geological data
- Dynamic filtering and visualization options
- Predictive interface for magnitude estimation
- Risk assessment visualization for major Turkish cities

### Technical Stack
- **Python**: Core programming language
- **Data Processing**: Pandas, NumPy, GeoPandas
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Deep Learning**: PyTorch with GPU acceleration
- **Visualization**: Matplotlib, Seaborn, Plotly, Folium
- **Web Application**: Streamlit
- **Version Control**: Git

## Results & Findings

### Supervised Learning
- The best-performing model was XGBoost with RMSE of 0.2309 and R² of 0.847
- Most important features included: multi-fault weighted risk, spatial density measures, fault complexity scores, and temporal patterns
- Multi-fault analysis significantly outperformed single nearest-fault approaches

### Unsupervised Learning
- K-Means clustering identified 5 distinct earthquake risk zones across Turkey
- DBSCAN revealed fine-grained density patterns and isolated high-activity zones
- Clear correlation between earthquake clusters and major fault systems
- High-risk zones concentrated along the North Anatolian Fault and high-population areas
- Cities with highest earthquake risk: Istanbul, Izmir, Van, Sakarya, Bursa

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
├── data/                  # Data files
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
1. **Unsupervised Learning**: Implemented K-Means and DBSCAN clustering on the earthquake dataset
2. **GPU Acceleration**: PyTorch-based neural network with CUDA support for magnitude prediction
3. **Web Application Deployment**: Interactive Streamlit application for exploring all analyses

## Real-World Applications & Impact

This project addresses several critical real-world problems:

### Emergency Management & Preparedness
- **Early Warning Systems**: The magnitude prediction model can estimate potential earthquake severity based on location and geological factors
- **Resource Allocation**: City risk assessment helps emergency management agencies prioritize resource distribution and infrastructure investments
- **Evacuation Planning**: Identified high-risk zones inform evacuation route planning and shelter positioning

### Urban Planning & Infrastructure Development
- **Building Code Requirements**: Risk zone identification guides region-specific building code enforcement
- **Critical Infrastructure Placement**: Hospitals, schools, and emergency response facilities can be strategically located based on seismic risk patterns
- **Insurance Risk Assessment**: Insurance companies can use the risk models to calibrate premiums and coverage in different regions

### Public Awareness & Education
- **Risk Communication**: Interactive visualizations effectively communicate complex risk patterns to the general public
- **Community Preparedness**: City-specific risk assessments drive targeted education campaigns

## Possible Future Enhancements

### Data Integration & Expansion
- **Real-time Data Pipeline**: Connect to AFAD's real-time earthquake API for live monitoring and prediction
- **Historical Damage Records**: Incorporate building damage and casualty data to predict potential impact beyond magnitude
- **Geological Data**: Integrate soil composition, liquefaction potential, and groundwater data for more accurate risk assessment

### Advanced Modeling Approaches
- **Time Series Forecasting**: Implement LSTM/GRU networks to capture temporal patterns and potential precursor events
- **Transfer Learning**: Apply models trained on global earthquake data to improve Turkey-specific predictions
- **Ensemble Methods**: Combine multiple prediction models for more robust forecasting
- **Bayesian Networks**: Implement probabilistic models that better represent uncertainty in predictions

### Technical Improvements
- **Mobile Application**: Develop a companion mobile app for on-the-go risk assessment and alerts
- **Microservices Architecture**: Refactor the application into containerized microservices for better scalability
- **Offline Capability**: Enable core functionality without internet connection for use during emergencies
- **Multi-language Support**: Add Turkish interface to improve accessibility for local users

### Research Extensions
- **Cross-Border Analysis**: Extend the model to neighboring countries for comprehensive regional assessment
- **Aftershock Prediction**: Develop specialized models for predicting aftershock patterns following major events

### Acknowledgments
* AFAD (Disaster and Emergency Management Presidency) for the earthquake catalog data
* Global AI Hub Machine Learning Bootcamp for the project opportunity
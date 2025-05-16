# components/clustering.py
# Path: /components/clustering.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def show_clustering(df):
    """Clustering visualization and analysis page"""
    st.title("Earthquake Cluster Analysis")
    
    # Check if clustering data is available
    has_kmeans = 'KMeans_Cluster' in df.columns
    has_dbscan = 'DBSCAN_Cluster' in df.columns
    
    if not (has_kmeans or has_dbscan):
        st.error("Clustering data not available in the dataset.")
        return
    
    # Select clustering method to visualize
    cluster_methods = []
    if has_kmeans:
        cluster_methods.append("K-Means")
    if has_dbscan:
        cluster_methods.append("DBSCAN")
    
    cluster_method = st.selectbox("Select Clustering Method", cluster_methods)
    
    if cluster_method == "K-Means":
        show_kmeans_analysis(df)
    elif cluster_method == "DBSCAN":
        show_dbscan_analysis(df)

def show_kmeans_analysis(df):
    """Show K-Means clustering analysis"""
    st.subheader("K-Means Clustering Analysis")
    
    # Cluster distribution
    st.write("### Cluster Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_counts = df['KMeans_Cluster'].value_counts().sort_index()
    cluster_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Earthquakes')
    ax.set_title('Distribution of Earthquakes Across K-Means Clusters')
    st.pyplot(fig)
    
    # Cluster characteristics
    st.write("### Cluster Characteristics")
    cluster_analysis = df.groupby('KMeans_Cluster').agg({
        'Longitude': 'mean',
        'Latitude': 'mean',
        'Depth': 'mean',
        'Magnitude': 'mean',
        'KMeans_Cluster': 'count'
    }).rename(columns={'KMeans_Cluster': 'Count'})
    
    # Add fault distance stats if available
    if 'distance_to_fault' in df.columns:
        fault_stats = df.groupby('KMeans_Cluster').agg({
            'distance_to_fault': ['mean', 'min'],
            'nearest_fault_importance': 'mean'
        })
        
        # Flatten MultiIndex columns
        fault_stats.columns = ['_'.join(col).strip() for col in fault_stats.columns.values]
        cluster_analysis = pd.concat([cluster_analysis, fault_stats], axis=1)
    
    st.dataframe(cluster_analysis.round(2))
    
    # Use pre-rendered 3D visualization
    st.write("### 3D Cluster Visualization")
    
    # Check for pre-rendered visualizations
    kmeans_map_path = "maps/kmeans_clusters_map_plotly.html"
    
    if os.path.exists(kmeans_map_path):
        st.components.v1.html(open(kmeans_map_path, 'r', encoding='utf-8').read(), height=600)
    else:
        st.error("Pre-rendered K-means visualization not found. Please run the notebook to generate it.")

def show_dbscan_analysis(df):
    """Show DBSCAN clustering analysis"""
    st.subheader("DBSCAN Clustering Analysis")
    
    # Count the number of clusters and noise points
    n_clusters = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'].values else 0)
    n_noise = (df['DBSCAN_Cluster'] == -1).sum()
    
    st.write(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points.")
    st.write(f"Percentage of noise points: {n_noise / len(df) * 100:.2f}%")
    
    # Cluster distribution
    st.write("### Cluster Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_counts = df['DBSCAN_Cluster'].value_counts().sort_index()
    cluster_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Earthquakes')
    ax.set_title('Distribution of Earthquakes Across DBSCAN Clusters')
    st.pyplot(fig)
    
    # Cluster characteristics (excluding noise)
    st.write("### Cluster Characteristics (excluding noise)")
    clean_df = df[df['DBSCAN_Cluster'] != -1]
    if len(clean_df) > 0:
        cluster_analysis = clean_df.groupby('DBSCAN_Cluster').agg({
            'Longitude': 'mean',
            'Latitude': 'mean',
            'Depth': 'mean',
            'Magnitude': 'mean',
            'DBSCAN_Cluster': 'count'
        }).rename(columns={'DBSCAN_Cluster': 'Count'})
        
        # Add fault distance stats if available
        if 'distance_to_fault' in df.columns:
            fault_stats = clean_df.groupby('DBSCAN_Cluster').agg({
                'distance_to_fault': ['mean', 'min'],
                'nearest_fault_importance': 'mean'
            })
            
            # Flatten MultiIndex columns
            fault_stats.columns = ['_'.join(col).strip() for col in fault_stats.columns.values]
            cluster_analysis = pd.concat([cluster_analysis, fault_stats], axis=1)
        
        st.dataframe(cluster_analysis.round(2))
    
    # Use pre-rendered 3D visualization
    st.write("### 3D Cluster Visualization")
    
    # Check for pre-rendered visualizations
    dbscan_map_path = "maps/dbscan_clusters_map_plotly.html"
    
    if os.path.exists(dbscan_map_path):
        st.components.v1.html(open(dbscan_map_path, 'r', encoding='utf-8').read(), height=600)
    else:
        st.error("Pre-rendered DBSCAN visualization not found. Please run the notebook to generate it.")
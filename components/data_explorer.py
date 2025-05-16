# components/data_explorer.py
# Path: /components/data_explorer.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def show_data_explorer(df):
    """Data exploration page"""
    st.title("Earthquake Data Explorer")
    
    # Basic dataset information
    st.subheader("Dataset Overview")
    st.write(f"Total records: {len(df)}")
    
    # Show sample data
    if st.checkbox("Show sample data"):
        st.dataframe(df.head(10))
    
    # Show column information
    if st.checkbox("Show column information"):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Missing Values': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(col_info)
    
    # Visualization section
    st.subheader("Visualizations")
    
    viz_type = st.selectbox(
        "Select Visualization",
        ["Magnitude Distribution", "Depth Distribution", "Temporal Analysis", 
         "Magnitude vs Depth", "Geographic Distribution"]
    )
    
    if viz_type == "Geographic Distribution":
        # Use plotly for interactive plotting - use entire dataset without sampling
        fig = px.scatter_mapbox(
            df,  # Use full dataset
            lat='Latitude', 
            lon='Longitude',
            color='Magnitude',
            size='Magnitude',
            color_continuous_scale='Viridis',
            size_max=15,
            zoom=5,
            center={"lat": 38.5, "lon": 35.5},
            mapbox_style="open-street-map",
            title='Geographic Distribution of Earthquakes',
            hover_data=['Depth', 'Magnitude']
        )
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig)
    
    elif viz_type == "Magnitude Distribution":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Magnitude'], bins=30, kde=True, ax=ax)
        ax.axvline(df['Magnitude'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {df["Magnitude"].mean():.2f}')
        ax.set_title('Distribution of Earthquake Magnitudes')
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)
        
        st.write(f"Minimum Magnitude: {df['Magnitude'].min():.2f}")
        st.write(f"Maximum Magnitude: {df['Magnitude'].max():.2f}")
        st.write(f"Mean Magnitude: {df['Magnitude'].mean():.2f}")
        st.write(f"Median Magnitude: {df['Magnitude'].median():.2f}")
    
    elif viz_type == "Depth Distribution":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Depth'], bins=30, kde=True, ax=ax)
        ax.axvline(df['Depth'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {df["Depth"].mean():.2f}')
        ax.set_title('Distribution of Earthquake Depths')
        ax.set_xlabel('Depth (km)')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)
        
        st.write(f"Minimum Depth: {df['Depth'].min():.2f} km")
        st.write(f"Maximum Depth: {df['Depth'].max():.2f} km")
        st.write(f"Mean Depth: {df['Depth'].mean():.2f} km")
        st.write(f"Median Depth: {df['Depth'].median():.2f} km")
    
    elif viz_type == "Temporal Analysis":
        # Check if Date column exists and is datetime
        if 'Date' in df.columns:
            # Try to convert to datetime if not already
            if df['Date'].dtype != 'datetime64[ns]':
                try:
                    # Try with explicit format first
                    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y %H:%M:%S", errors='coerce')
                    
                    # Check if any dates couldn't be parsed
                    null_dates = df['Date'].isnull().sum()
                    if null_dates > 0:
                        # Try alternative formats
                        df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y %H:%M:%S", errors='coerce')
                        if df['Date'].isnull().sum() > 0:
                            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                except:
                    st.error("Could not convert Date column to datetime format.")
                    return
            
            # Extract year if not present
            if 'Year' not in df.columns:
                df['Year'] = df['Date'].dt.year
            
            # Yearly analysis
            yearly_counts = df.groupby('Year').size()
            fig, ax = plt.subplots(figsize=(12, 6))
            yearly_counts.plot(kind='bar', ax=ax)
            ax.set_title('Yearly Earthquake Frequency')
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of Earthquakes')
            st.pyplot(fig)
            
            # Monthly analysis if available
            if 'Month' not in df.columns and 'Date' in df.columns:
                df['Month'] = df['Date'].dt.month
            
            if 'Month' in df.columns:
                monthly_counts = df.groupby('Month').size()
                fig, ax = plt.subplots(figsize=(10, 6))
                monthly_counts.plot(kind='bar', ax=ax)
                ax.set_title('Monthly Earthquake Frequency')
                ax.set_xlabel('Month')
                ax.set_ylabel('Number of Earthquakes')
                ax.set_xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                st.pyplot(fig)
        else:
            st.error("Date column not found in the dataset.")
    
    elif viz_type == "Magnitude vs Depth":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Depth', y='Magnitude', data=df, alpha=0.6, ax=ax)
        ax.set_title('Relationship Between Earthquake Depth and Magnitude')
        ax.set_xlabel('Depth (km)')
        ax.set_ylabel('Magnitude')
        st.pyplot(fig)
        
        # Calculate correlation
        correlation = df['Depth'].corr(df['Magnitude'])
        st.write(f"Correlation between Depth and Magnitude: {correlation:.4f}")
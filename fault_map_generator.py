# fault_map_generator.py
import geopandas as gpd
import folium
import os

# Create maps directory if it doesn't exist
os.makedirs('maps', exist_ok=True)

# Load fault line data
fault_gdf = gpd.read_file('data/tr_faults_imp.geojson')
print(f"Number of fault lines loaded: {len(fault_gdf)}")

# Create a base map centered on Turkey using OpenStreetMap
turkey_map = folium.Map(location=[38.5, 35.5], zoom_start=6, tiles="OpenStreetMap")

# Define importance categories
importance_ranges = {
    'High': [4, 5],
    'Medium': [3],
    'Low': [1, 2]
}

# Create separate feature groups for different importance levels
groups = {
    'High': folium.FeatureGroup(name="High Importance Faults", show=True),
    'Medium': folium.FeatureGroup(name="Medium Importance Faults", show=True),
    'Low': folium.FeatureGroup(name="Low Importance Faults", show=True)
}

# Add each fault line individually to avoid GeoJSON structure issues
for _, fault in fault_gdf.iterrows():
    importance = fault.get('importance', 1)
    
    # Determine category and color
    category = 'Low'
    if importance >= 4:
        category = 'High'
        color = '#FF0000'  # Red
    elif importance >= 3:
        category = 'Medium'
        color = '#FFA500'  # Orange
    else:
        color = '#FFFF00'  # Yellow
    
    # Add as a simple polyline instead of GeoJSON
    try:
        # Extract coordinates
        if fault.geometry.geom_type == 'LineString':
            coords = [(y, x) for x, y in fault.geometry.coords]
            folium.PolyLine(
                coords,
                color=color,
                weight=importance*0.5,
                opacity=0.7,
                tooltip=f"Fault: {fault.get('FAULT_NAME', 'Unknown')}, Importance: {importance}"
            ).add_to(groups[category])
        elif fault.geometry.geom_type == 'MultiLineString':
            for line in fault.geometry.geoms:
                coords = [(y, x) for x, y in line.coords]
                folium.PolyLine(
                    coords, 
                    color=color,
                    weight=importance*0.5,
                    opacity=0.7,
                    tooltip=f"Fault: {fault.get('FAULT_NAME', 'Unknown')}, Importance: {importance}"
                ).add_to(groups[category])
    except Exception as e:
        print(f"Error processing fault {fault.get('FAULT_NAME', 'Unknown')}: {str(e)}")

# Add all feature groups to the map
for group in groups.values():
    group.add_to(turkey_map)

# Add a legend
legend_html = '''
<div style="position: fixed; bottom: 50px; right: 50px; width: 200px; 
    background-color: white; border:2px solid grey; z-index:9999; 
    padding: 10px; border-radius: 5px; font-family: Arial;">
    <p><b>Fault Line Importance:</b></p>
    <p><span style="color:#FF0000;">━━━</span> High (4+)</p>
    <p><span style="color:#FFA500;">━━━</span> Medium (3)</p>
    <p><span style="color:#FFFF00;">━━━</span> Low (<3)</p>
</div>
'''
turkey_map.get_root().html.add_child(folium.Element(legend_html))

# Add layer control
folium.LayerControl().add_to(turkey_map)

# Save map to file
output_file = 'maps/turkey_faults_map.html'
turkey_map.save(output_file)
print(f"Map saved to {output_file}")
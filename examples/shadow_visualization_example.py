#!/usr/bin/env python3
"""
Example script showing how to use the shadow visualization functionality.

This example demonstrates how to create comprehensive shadow analysis plots
for buildings with inference results containing shadow casting data.
"""
import hypercadaster_ES as hc
import pandas as pd

def main():
    """Run shadow visualization example."""
    
    print("🏢 Shadow Visualization Example")
    print("="*40)
    
    # Example 1: Load results from a building analysis with shadow data
    print("\n1. Loading building data with shadow analysis...")
    
    # In a real scenario, you would load results from hc.merge() with building_parts_inference=True
    # For this example, we'll create sample data matching the expected format
    
    sample_building_data = {
        'building_reference': '6161378DF2766A',
        'building_use': 'Residential',
        'year_of_construction': '1995',
        'elevation': 45.2,
        'br__number_of_floors': 3,
        'br__shadows_at_distance': {
            # Distance from building edge to closest neighboring building
            # 0° = North, 90° = East, 180° = South, 270° = West
            '0': [25.5, 52.3, 79.1],     # North - moderate shadows
            '30': [15.2, 31.8, 48.4],   # Northeast - shorter shadows  
            '60': [12.1, 25.3, 38.5],   # East-northeast - short shadows
            '90': [8.7, 18.2, 27.7],    # East - minimal shadows
            '120': [18.4, 38.6, 58.8],  # Southeast - growing shadows
            '150': [35.2, 73.9, 112.6], # South-southeast - longer shadows
            '180': [96.4, 202.1, 307.8], # South - longest shadows (midday sun from north)
            '210': [42.3, 88.7, 135.1], # Southwest - long shadows
            '240': [28.6, 60.0, 91.4],  # West-southwest - moderate shadows
            '270': [19.8, 41.5, 63.2],  # West - shorter shadows
            '300': [22.1, 46.3, 70.5],  # Northwest - moderate shadows
            '330': [28.9, 60.6, 92.3]   # North-northwest - growing shadows
        },
        'br__building_contour_at_distance': {
            # Distance from building centroid to building edge
            '0': 45.2,    # North side
            '30': 52.1,   # Northeast corner
            '60': 38.7,   # East side
            '90': 41.3,   # East side  
            '120': 48.9,  # Southeast corner
            '150': 55.6,  # South side
            '180': 62.3,  # South side
            '210': 58.1,  # Southwest corner
            '240': 44.8,  # West side
            '270': 42.5,  # West side
            '300': 47.2,  # Northwest corner
            '330': 49.8   # North side
        },
        'br__elevation_at_shadow': {
            # NEW: Elevation at closest neighbor building by orientation (in meters)
            '0': 48.5,    # North - neighbor at 48.5m elevation
            '30': 52.8,   # Northeast - neighbor at 52.8m elevation
            '60': 45.1,   # East - neighbor at 45.1m elevation
            '90': 47.6,   # East - neighbor at 47.6m elevation
            '120': 51.3,  # Southeast - neighbor at 51.3m elevation
            '150': 49.7,  # South - neighbor at 49.7m elevation
            '180': 53.2,  # South - neighbor at 53.2m elevation (tallest)
            '210': 50.4,  # Southwest - neighbor at 50.4m elevation
            '240': 46.9,  # West - neighbor at 46.9m elevation
            '270': 44.8,  # West - neighbor at 44.8m elevation
            '300': 47.1,  # Northwest - neighbor at 47.1m elevation
            '330': 49.2   # North - neighbor at 49.2m elevation
        }
    }
    
    # Create DataFrame (in real usage, this would come from hc.merge())
    building_gdf = pd.DataFrame([sample_building_data])
    
    print(f"   Building: {sample_building_data['building_reference']}")
    print(f"   Elevation: {sample_building_data['elevation']}m")
    print(f"   Floors: {sample_building_data['br__number_of_floors']}")
    
    # Example 2: Create shadow analysis visualization
    print("\n2. Creating shadow analysis visualization...")
    
    try:
        # Create interactive plot (will fallback to matplotlib if plotly unavailable)
        # Floor height is automatically detected from the data
        fig = hc.plot_building_shadows_analysis(
            gdf=building_gdf,
            building_reference='6161378DF2866A',
            output_path='shadow_analysis_example.pdf',  # Save as PDF
            interactive=True,                           # Try interactive first
            include_elevation=True                      # Include elevation effects
        )
        
        print("   ✅ Shadow analysis visualization created!")
        print("   📄 Saved as: shadow_analysis_example.pdf")
        print("   📊 Contains:")
        print("      - Polar plot showing shadow distances by orientation")
        print("      - Linear plot comparing building contour vs shadows") 
        print("      - Multi-floor shadow comparison")
        print("      - Building height profile with elevation")
        
    except Exception as e:
        print(f"   ❌ Error creating visualization: {e}")
    
    # Example 3: Analyze shadow patterns
    print("\n3. Analyzing shadow patterns...")
    
    shadow_data = building_gdf.iloc[0]['br__shadows_at_distance']
    
    # Find orientation with longest shadows (first floor)
    max_shadow_orientation = None
    max_shadow_distance = 0
    
    for orientation, shadow_list in shadow_data.items():
        if isinstance(shadow_list, list) and len(shadow_list) > 0:
            shadow_distance = shadow_list[0]  # First floor
            if shadow_distance > max_shadow_distance:
                max_shadow_distance = shadow_distance
                max_shadow_orientation = orientation
    
    print(f"   🎯 Longest shadow: {max_shadow_distance:.1f}m towards {max_shadow_orientation}°")
    
    # Find orientation with shortest shadows
    min_shadow_orientation = None
    min_shadow_distance = float('inf')
    
    for orientation, shadow_list in shadow_data.items():
        if isinstance(shadow_list, list) and len(shadow_list) > 0:
            shadow_distance = shadow_list[0]  # First floor
            if shadow_distance < min_shadow_distance:
                min_shadow_distance = shadow_distance
                min_shadow_orientation = orientation
                
    print(f"   🎯 Shortest shadow: {min_shadow_distance:.1f}m towards {min_shadow_orientation}°")
    
    # Calculate shadow efficiency (building area vs shadow impact)
    contour_data = building_gdf.iloc[0]['br__building_contour_at_distance']
    avg_building_radius = sum(contour_data.values()) / len(contour_data)
    avg_shadow_distance = sum([shadow_list[0] for shadow_list in shadow_data.values() 
                              if isinstance(shadow_list, list) and len(shadow_list) > 0]) / len(shadow_data)
    
    shadow_ratio = avg_shadow_distance / avg_building_radius
    print(f"   📏 Average building radius: {avg_building_radius:.1f}m")
    print(f"   📏 Average shadow distance: {avg_shadow_distance:.1f}m") 
    print(f"   📊 Shadow-to-building ratio: {shadow_ratio:.2f}")
    
    # NEW: Analyze elevation at closest neighbors
    print("\n   🏔️  Neighbor elevation analysis:")
    elevation_data = building_gdf.iloc[0]['br__elevation_at_shadow']
    
    # Find highest and lowest neighbor elevations
    max_elevation = max(elevation_data.values())
    min_elevation = min(elevation_data.values())
    avg_elevation = sum(elevation_data.values()) / len(elevation_data)
    
    max_elevation_orientation = [k for k, v in elevation_data.items() if v == max_elevation][0]
    min_elevation_orientation = [k for k, v in elevation_data.items() if v == min_elevation][0]
    
    print(f"   📐 Highest neighbor: {max_elevation:.1f}m at {max_elevation_orientation}°")
    print(f"   📐 Lowest neighbor: {min_elevation:.1f}m at {min_elevation_orientation}°")  
    print(f"   📐 Average neighbor elevation: {avg_elevation:.1f}m")
    print(f"   📐 Elevation difference: {max_elevation - min_elevation:.1f}m")
    
    print("\n4. Usage in real projects:")
    print("   💡 To use with actual data:")
    print("      gdf = hc.merge(wd, cadaster_codes=['08900'], building_parts_inference=True)")
    print("      hc.plot_building_shadows_analysis(gdf, 'YOUR_BUILDING_REF')")
    print("   💡 New elevation indicator available:")
    print("      br__elevation_at_shadow: elevation of closest neighbor by orientation")
    print("   💡 For multiple buildings, iterate through unique building_reference values")
    print("   💡 Combine with solar analysis for energy efficiency studies")
    print("   💡 Use for urban planning to minimize shadow impact on neighbors")
    print("   💡 Use elevation data for slope analysis and drainage studies")

if __name__ == "__main__":
    main()
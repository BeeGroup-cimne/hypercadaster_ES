# Configuration & Examples

This document provides comprehensive configuration options and practical examples for using hypercadaster_ES effectively across different scenarios.

## ⚙️ Advanced Configuration & Customization

### Geographic Area Selection Methods

#### By Administrative Levels

```python
import hypercadaster_ES as hc

# Spanish provinces (2-digit codes) - Complete province analysis
hc.download("./data", province_codes=["08", "28", "46"])  # Barcelona, Madrid, Valencia

# Municipality INE codes (5-digit codes) - Specific cities
hc.download("./data", ine_codes=["08019", "28079", "46250"])  # Barcelona, Madrid, Valencia cities

# Cadastral codes - Direct cadastral reference
hc.download("./data", cadaster_codes=["08900", "28900", "46250"])  # Barcelona, Madrid, Valencia cities
```

#### **New: Flexible Parameter Combination**

You can now combine different geographic parameters in the same call:

```python
# ✅ Combine province, INE codes, and cadaster codes
hc.download(
    "./data", 
    province_codes=["25"],              # Entire Lleida province
    ine_codes=["08019"],               # Barcelona city
    cadaster_codes=["46250"]          # Valencia city
)

# ✅ Mix single values and lists (auto-converted)
hc.download(
    "./data",
    province_codes="08",               # Single province (auto-converted to list)
    cadaster_codes=["25023", "25285"] # Multiple specific municipalities
)

# ✅ All parameters are automatically deduplicated
hc.download(
    "./data",
    province_codes=["25"],             # Includes municipality 25023
    cadaster_codes=["25023"]          # Duplicate - automatically removed
)
```

### Comprehensive Data Layer Configuration

#### Basic Data Layers

```python
hc.download(
    "./data",
    province_codes=["08"],
    neighborhood_layer=True,        # Barcelona neighborhood boundaries
    postal_code_layer=True,         # Spanish postal code boundaries
    census_layer=True,             # INE census tract data
    elevation_layer=True,          # Digital elevation model
    open_data_layers=True,         # Barcelona open data integration
    force=False                    # Skip download if files exist
)
```

#### Advanced Analysis Configuration

```python
gdf = hc.merge(
    "./data",
    cadaster_codes=["25023"],   # Alpicat municipality
    
    # Basic data integration
    neighborhood_layer=True,
    postal_code_layer=True,
    census_layer=True,
    elevations_layer=True,
    open_data_layers=True,
    
    # Advanced building analysis
    building_parts_inference=True,     # Enable geometric analysis
    building_parts_plots=False,        # Generate visualization plots (only for small areas)
    
    # CAT file integration (requires separate download)
    use_CAT_files=True,               # Use detailed building space data
    CAT_files_rel_dir="CAT_files"     # CAT files directory location
)
```

### CAT Files Integration

CAT files provide the most detailed building information available from Spanish cadastral sources.

#### Downloading CAT Files

Visit the [Spanish Cadastral Service](https://www.sedecatastro.gob.es/Accesos/SECAccDescargaDatos.aspx) portal:

1. Click "Descarga de información alfanumérica por provincia (formato CAT)"
2. Select your province of interest
3. Download the provincial CAT files
4. Extract to `CAT_files/` directory in your working directory

#### CAT File Directory Structure

```
data_directory/
├── cadaster/
├── CAT_files/
│   ├── 08279U_17012025.zip
│   ├── 08900U_17012025.zip
│   └── ...
└── ...
```

#### Using CAT Files

```python
# Ensure CAT files are available in the specified directory
gdf = hc.merge(
    "./data",
    cadaster_codes=["08900"],
    use_CAT_files=True,
    CAT_files_rel_dir="CAT_files"  # Relative to working directory
)
```

## 🚀 Practical Examples

### Example 1: Basic Municipal Analysis

Simple analysis for a single municipality without advanced inference:

```python
import hypercadaster_ES as hc

# Configuration
wd = "./municipal_analysis"
municipality = ["46250"]  # Valencia

# Download basic data
hc.download(
    wd=wd,
    cadaster_codes=municipality,
    postal_code_layer=True,
    census_layer=True
)

# Process data
gdf = hc.merge(
    wd=wd,
    cadaster_codes=municipality,
    postal_code_layer=True,
    census_layer=True,
    building_parts_inference=False  # Disable for faster processing
)

# Save results
gdf.to_pickle(f"{wd}/valencia_basic.pkl", compression="gzip")
print(f"Processed {len(gdf)} buildings")
```

### Example 2: Advanced Building Analysis

Comprehensive analysis with full inference capabilities:

```python
import hypercadaster_ES as hc

# Configuration for detailed analysis
wd = "./advanced_analysis"
municipalities = ["25023", "25285"]  # Alpicat, Seròs (Lleida)

# Download all available data layers
hc.download(
    wd=wd,
    cadaster_codes=municipalities,
    neighborhood_layer=True,
    postal_code_layer=True,
    census_layer=True,
    elevation_layer=True,
    open_data_layers=True
)

# Perform comprehensive analysis
gdf = hc.merge(
    wd=wd,
    cadaster_codes=municipalities,
    neighborhood_layer=True,
    postal_code_layer=True,
    census_layer=True,
    elevations_layer=True,
    open_data_layers=True,
    building_parts_inference=True,    # Enable advanced analysis
    building_parts_plots=True,        # Generate plots (small area)
    use_CAT_files=True,
    CAT_files_rel_dir="CAT_files"
)

# Save with descriptive filename
output_file = f"{wd}/{'_'.join(municipalities)}_complete.pkl"
gdf.to_pickle(output_file, compression="gzip")

print(f"Advanced analysis completed:")
print(f"  - Buildings processed: {len(gdf)}")
print(f"  - Output saved to: {output_file}")
```

### Example 3: Provincial-Scale Analysis

Large-scale analysis optimized for performance:

```python
import hypercadaster_ES as hc

# Configuration for provincial analysis
wd = "./provincial_analysis"
province = ["25"]  # Lleida province

# Download data (may take hours)
print("Starting provincial download - this may take several hours...")
hc.download(
    wd=wd,
    province_codes=province,
    census_layer=True,
    elevation_layer=True
)

# Process with performance optimizations
print("Processing provincial data...")
gdf = hc.merge(
    wd=wd,
    province_codes=province,
    census_layer=True,
    elevations_layer=True,
    building_parts_inference=False,   # Disable for memory efficiency
    building_parts_plots=False,       # Disable plotting
    use_CAT_files=False              # Disable for faster processing
)

# Save results
output_file = f"{wd}/lleida_province.pkl"
gdf.to_pickle(output_file, compression="gzip")

print(f"Provincial analysis completed:")
print(f"  - Buildings processed: {len(gdf):,}")
print(f"  - Output saved to: {output_file}")
```

### Example 4: Barcelona Enhanced Analysis

Taking advantage of Barcelona's enhanced open data layers:

```python
import hypercadaster_ES as hc

# Barcelona with all enhancements
wd = "./barcelona_enhanced"
barcelona = ["08900"]

# Download with Barcelona-specific layers
hc.download(
    wd=wd,
    cadaster_codes=barcelona,
    neighborhood_layer=True,        # Barcelona neighborhoods
    postal_code_layer=True,
    census_layer=True,
    elevation_layer=True,
    open_data_layers=True          # Barcelona commercial data
)

# Process with full Barcelona features
gdf = hc.merge(
    wd=wd,
    cadaster_codes=barcelona,
    neighborhood_layer=True,
    postal_code_layer=True,
    census_layer=True,
    elevations_layer=True,
    open_data_layers=True,
    building_parts_inference=True,
    building_parts_plots=False,    # Too large for plots
    use_CAT_files=True,
    CAT_files_rel_dir="CAT_files"
)

# Save Barcelona analysis
output_file = f"{wd}/barcelona_complete.pkl"
gdf.to_pickle(output_file, compression="gzip")

print(f"Barcelona enhanced analysis:")
print(f"  - Buildings: {len(gdf):,}")
print(f"  - Neighborhoods: {gdf['neighborhood_name'].nunique()}")
print(f"  - Districts: {gdf['district_name'].nunique()}")
```

### Example 5: Mixed Geographic Selection

**New feature**: Combine different geographic parameters for flexible analysis:

```python
import hypercadaster_ES as hc

# Mixed geographic analysis - combine province, specific cities, and rural areas
wd = "./mixed_geographic_analysis"

# ✅ Combine multiple geographic parameter types
hc.download(
    wd=wd,
    province_codes="25",                    # Entire Lleida province (auto-converted to list)
    ine_codes=["08019", "46250"],          # Barcelona and Valencia cities
    cadaster_codes=["03014", "11012"],     # Alicante and Cádiz (additional municipalities)
    
    # Full data layers for comprehensive analysis
    postal_code_layer=True,
    census_layer=True,
    elevation_layer=True,
    open_data_layers=True
)

# Process the combined geographic area
gdf = hc.merge(
    wd=wd,
    # Same parameters are automatically combined and deduplicated
    province_codes="25",                    # Redundant with individual municipalities - auto-deduplicated
    ine_codes=["08019", "46250"],          
    cadaster_codes=["03014", "11012", "25023"],  # 25023 is in Lleida province - auto-deduplicated
    
    postal_code_layer=True,
    census_layer=True,
    elevations_layer=True,
    open_data_layers=True,
    building_parts_inference=True,
    use_CAT_files=True,
    CAT_files_rel_dir="CAT_files"
)

# Save results with geographic identifier
output_file = f"{wd}/mixed_geographic_analysis.pkl"
gdf.to_pickle(output_file, compression="gzip")

print(f"Mixed geographic analysis completed:")
print(f"  - Total municipalities processed: {gdf['municipality_ine_code'].nunique()}")
print(f"  - Total buildings: {len(gdf):,}")
print(f"  - Provinces covered: {gdf['province_code'].nunique()}")
print(f"  - Output saved to: {output_file}")
```

## 🔧 Performance & Memory Optimization

### Large-Scale Processing Configuration

For provincial or multi-municipal analysis:

```python
# Memory-efficient configuration
gdf = hc.merge(
    "./data",
    province_codes=["08"],
    building_parts_inference=False,    # Disable for memory efficiency
    building_parts_plots=False,        # Disable for speed
    use_CAT_files=False                # Disable for faster processing
)
```

### Memory-Efficient Municipality Processing

Process large cities individually to manage memory usage:

```python
# Process municipalities individually for large cities
municipalities = ["08019", "08096", "08121"]  # Barcelona area

for ine_code in municipalities:
    print(f"Processing municipality: {ine_code}")
    
    # Download data for individual municipality
    hc.download(f"./data_{ine_code}", ine_codes=[ine_code])
    
    # Process individual municipality
    gdf_municipal = hc.merge(
        f"./data_{ine_code}",
        ine_codes=[ine_code],
        building_parts_inference=True
    )
    
    # Save individual result
    gdf_municipal.to_pickle(f"results_{ine_code}.pkl", compression="gzip")
    
    print(f"  - Completed: {len(gdf_municipal):,} buildings")
```

### Batch Processing Multiple Areas

Process multiple small municipalities efficiently:

```python
# Batch processing for multiple small municipalities
municipalities = ["25023", "25285", "25120", "25138"]  # Lleida area

all_results = []

for muni in municipalities:
    print(f"Processing {muni}...")
    
    # Download and process
    hc.download(f"./batch_data", cadaster_codes=[muni], force=False)
    
    gdf = hc.merge(
        "./batch_data",
        cadaster_codes=[muni],
        building_parts_inference=True,
        building_parts_plots=False
    )
    
    # Add municipality identifier
    gdf['processing_municipality'] = muni
    all_results.append(gdf)

# Combine all results
import pandas as pd
combined_gdf = pd.concat(all_results, ignore_index=True)
combined_gdf.to_pickle("./batch_results.pkl", compression="gzip")

print(f"Batch processing completed: {len(combined_gdf):,} total buildings")
```

## 📊 Configuration Reference

### Complete Parameter Reference

```python
# Download function parameters
hc.download(
    wd="./data",                    # Working directory (required)
    
    # Geographic Area Selection - ✅ NEW: Can combine multiple parameters
    province_codes=None,            # String or list of 2-digit province codes
    ine_codes=None,                 # String or list of 5-digit INE municipality codes  
    cadaster_codes=None,            # String or list of cadastral municipality codes
    # Note: All geographic parameters are auto-converted to lists and can be combined
    
    # Data Layer Configuration
    neighborhood_layer=False,       # Include Barcelona neighborhoods
    postal_code_layer=False,        # Include postal code boundaries
    census_layer=False,             # Include INE census tracts
    elevation_layer=False,          # Include digital elevation model
    open_data_layers=False,         # Include Barcelona open data
    force=False                     # Force re-download existing files
)

# Merge function parameters
gdf = hc.merge(
    wd="./data",                    # Working directory (required)
    
    # Geographic Area Selection - ✅ NEW: Can combine multiple parameters
    province_codes=None,            # Same flexible combination as download function
    ine_codes=None,
    cadaster_codes=None,
    
    # Data Layer Configuration
    neighborhood_layer=False,       # Data layers (same as download)
    postal_code_layer=False,
    census_layer=False,
    elevations_layer=False,         # Note: 'elevations' vs 'elevation'
    open_data_layers=False,
    
    # Advanced Processing Options
    building_parts_inference=False, # Enable advanced building analysis (single municipality only)
    building_parts_plots=False,     # Generate visualization plots (single municipality only)  
    plot_zones_ratio=0.01,          # Ratio of zones to plot when building_parts_plots=True (0.01 = 1%)
    use_CAT_files=False,           # Use detailed CAT format files
    CAT_files_rel_dir="CAT_files"  # CAT files directory path
)
```

## 🎯 Best Practices

### Workflow Recommendations

1. **Start Small**: Begin with a single municipality to understand the data structure
2. **Test Configuration**: Verify your setup with `building_parts_inference=False` first
3. **Monitor Resources**: Check memory usage for large areas
4. **Save Intermediate Results**: Use compressed pickle files to save processing time
5. **Document Your Analysis**: Keep track of configuration parameters used

### Common Pitfalls to Avoid

- **❌ Multiple Municipalities + Building Analysis**: Never enable `building_parts_inference=True` or `building_parts_plots=True` with multiple municipalities - this will raise an error
- **Memory Issues**: Don't enable full inference for entire provinces without adequate RAM
- **Missing CAT Files**: Ensure CAT files are properly downloaded and placed
- **Path Issues**: Use absolute paths for working directories when possible
- **Network Timeouts**: Allow extra time for large downloads

### ⚠️ **Important Constraints**

**Building Analysis Limitation**: `building_parts_inference` and `building_parts_plots` can only be enabled for **single municipality** processing:

```python
# ✅ CORRECT: Single municipality with building analysis
hc.merge("./data", cadaster_codes=["25023"], building_parts_inference=True)

# ❌ ERROR: Multiple municipalities with building analysis  
hc.merge("./data", cadaster_codes=["25023", "25285"], building_parts_inference=True)  # Raises ValueError

# ❌ ERROR: Province-level with building analysis
hc.merge("./data", province_codes=["25"], building_parts_inference=True)  # Raises ValueError
```

**Workaround for Multiple Municipalities**:
```python
# Process municipalities individually
municipalities = ["25023", "25285", "25120"]
for muni in municipalities:
    gdf = hc.merge(
        f"./data_{muni}", 
        cadaster_codes=[muni], 
        building_parts_inference=True  # ✅ OK: Single municipality
    )
    gdf.to_pickle(f"results_{muni}.pkl")
```

---

[← Use Cases & Applications](use-cases-applications.md) | [Output Data Schema →](output-schema.md) | [Back to README](../README.md)
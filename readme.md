# T-Route Modifications: Flowpath-Level Lateral Flow Support

## T-Route - Tree-Based Channel Routing
To know more about T-Route, please click this link: https://github.com/CIROH-UA/t-route/blob/ngiab/readme.md

## Overview

This document describes modifications made to T-Route to enable reading catchment files directly from the output of hydrological models (e.g., CFE, NoahOWP) in NextGen in a Box. These modifications allow T-Route to provide lateral flow to every flowpath of a basin, perform routing for every flowpath, and enable proper gage mapping for data assimilation at the flowpath level.

## Background

### Previous Implementation
- T-Route could only read nexus files
- Each nexus contained cumulative lateral flow of one or multiple flowpaths
- Routing was performed for only one of the flowpaths associated with each nexus
- Streamflow gage mapping for data assimilation used hydrosequence-based downstream filtering from network data

### New Implementation
- T-Route can now read individual catchment files (`cat-*` files)
- Each catchment file provides lateral flow data for a specific flowpath
- Routing is performed for every segment in the basin
- Lateral flows are properly scaled based on catchment area and time units
- Streamflow gage mapping is derived directly from flowpaths data, enabling proper gage assignment for data assimilation at the flowpath level

## Modified Files

### 1. HYfeaturesNetwork.py

**File Path:** `src/troute-network/troute/HYFeaturesNetwork.py`

#### Modification 1: Flowpath ID Extraction and Unit Factor Calculation

**Function:** `build_qlateral_array()`

Added preprocessing of flowpath data to extract numeric IDs and calculate unit conversion factors based on catchment area.

```python
# Make a copy of the raw input DataFrame to avoid modifying the original data in-place
flowpath_dataframe = self._raw_dataframe.copy()

# Extract the numeric part of each flowpath ID (e.g., from "wb-1091162" to "1091162")
# Assumes 'id' column contains strings with a hyphen separator
flowpath_dataframe['id'] = [
    r.split('-')[1] if '-' in r else r 
    for r in flowpath_dataframe['id']
]

# Calculate a unit conversion factor (m²/s) based on area in km²
# Converts from km² to m² (×1,000²), then normalizes over an hourly time base (÷3600)
flowpath_dataframe['unit_factor'] = flowpath_dataframe['areasqkm'] * ((1000 ** 2) / 3600)

# Create a lookup dictionary that maps each flowpath ID to its corresponding unit factor
unit_factor = dict(
    zip(flowpath_dataframe['id'], flowpath_dataframe['unit_factor'])
)
```

**Purpose:**
- Extract numeric flowpath IDs from formatted strings (e.g., "wb-1091162" → "1091162")
- Create a lookup dictionary for efficient scaling of lateral flows

#### Modification 2: Catchment File Processing

**Function:** `build_qlateral_array()`

Added a new conditional branch to handle `cat-*` file pattern with parallel processing and unit scaling.

```python
elif qlat_file_pattern_filter == "cat-*":
    def process_file(f):
        df = pd.read_csv(f, usecols=['Time', 'Q_OUT'])
        df.rename(columns={'Time': 'timestamp', 'Q_OUT': 'qlat'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y%m%d%H%M')
        df = df.set_index('timestamp')
        df = df.T
        df.index = [int(os.path.basename(f).split('cat-')[1].split('_')[0].split('.')[0])]
        df = df.rename_axis(None, axis=1)
        df.index.name = 'feature_id'
        return df

    # Process all "cat-*" files in parallel
    with Parallel(n_jobs=-1) as p:
        dfs = p(delayed(process_file)(f) for f in qlat_files)

    # Concatenate all processed DataFrames into one
    nexuses_lateralflows_df = pd.concat(dfs, axis=0)

    # Apply unit scaling only in the 'cat-*' case
    nexuses_lateralflows_df = scale_lateral_flows(nexuses_lateralflows_df, unit_factor)

    qlats_df = nexuses_lateralflows_df[nexuses_lateralflows_df.index.isin(self.segment_index)]
```

**Purpose:**
- Extract flowpath ID from filename (e.g., `cat-1091162.csv` → 1091162)
- Transpose data to create feature_id-indexed DataFrame
- Apply unit scaling to convert flows to proper units
- Filter to include only segments present in the network

#### Modification 3: Unit Scaling Function

**Function:** `scale_lateral_flows()` (new function)

Added a new function to scale lateral flows based on catchment-specific unit factors.

```python
def scale_lateral_flows(df, unit_dict):
    """
    Multiply each row in the DataFrame by its corresponding unit factor.
    
    Parameters:
    - df: DataFrame with flowpath IDs as index
    - unit_dict: dict mapping flowpath ID (str or int) to scaling factor
    
    Returns:
    - A new DataFrame with selected rows scaled
    """
    # Convert all unit factor keys to integers
    factors = {int(k): v for k, v in unit_dict.items()}
    
    # Make a copy so the original DataFrame is not changed
    scaled_df = df.copy()
    
    # Loop through each key and apply the factor if key exists
    for key, factor in factors.items():
        if key in scaled_df.index:
            scaled_df.loc[key] = scaled_df.loc[key] * factor
        else:
            print(f"{key} not found in the DataFrame.")
    
    return scaled_df
```

**Purpose:**
- Apply catchment-specific unit conversion factors to lateral flow data
- Ensure proper units for routing calculations (m³/s)

#### Modification 4: Data Assimilation Gage Mapping

**Function:** `preprocess_data_assimilation()`

Modified the `preprocess_data_assimilation` function to derive streamflow gage mapping directly from the flowpaths dataframe instead of using hydrosequence-based filtering from the network dataframe.

```python
# Modification:
# Previously, streamflow DA gages were derived from `network`
# using hydrosequence-based downstream filtering.
# Now:
# Streamflow gage mapping is derived directly from `flowpaths[['id','gage']]`.

gages_df2 = flowpaths[['id', 'gage']].drop_duplicates()

# Remove missing gage assignments
gages_df2 = gages_df2[~gages_df2['gage'].isnull()]

# Convert flowpath IDs to integer segment IDs
gages_df2['id'] = (
    gages_df2['id']
    .str.split('-', expand=True)
    .loc[:, 1]
    .astype(float)
    .astype(int)
)

# Standardize column name to match network-based dataframe
gages_df2.rename(columns={'gage': 'value'}, inplace=True)

# Expand multi-gage entries
gages_df2['value'] = gages_df2.value.str.split(' ')
gages_df2 = (
    gages_df2
    .explode(column='value')
    .set_index('id')
    .join(
        pd.DataFrame()
        .from_dict(self.waterbody_connections, orient='index', columns=['lake_id'])
    )
)

# Identify USGS numeric gage IDs (used for streamflow DA)
usgs_ind2 = gages_df2.value.str.isnumeric()

idx_id2 = gages_df2.index.name
if not idx_id2:
    idx_id2 = 'index'
    
# NOTE:
# This version does NOT apply hydrosequence-based downstream filtering.
# It assumes gage-to-segment association is already topologically correct
# in the flowpaths input.

self._gages = (
    gages_df2.loc[usgs_ind2]
    .reset_index()
    .set_index(idx_id2)[['value']]
    .rename(columns={'value': 'gages'})
    .rename_axis(None, axis=0)
    .to_dict()
)
```

**Purpose:**
- Derive gage-to-segment mapping directly from flowpaths data
- Remove dependency on hydrosequence-based downstream filtering
- Assume gage-to-segment associations are already topologically correct in the input data
- Enable proper gage assignment for data assimilation at the flowpath level

### 2. AbstractNetwork.py

**File Path:** `src/troute-network/troute/AbstractNetwork.py`

#### Modification: Catchment File Discovery and Configuration

**Function:** `build_forcing_sets()`

Added support for discovering and configuring catchment files for model runs.

```python
elif forcing_glob_filter == "cat-*":
    # Get all files matching the 'cat-*' pattern, sorted alphabetically
    all_files = sorted(qlat_input_folder.glob(forcing_glob_filter))
    
    # Read the last timestamp from the 'Time' column of the first file
    final_timestamp_str = pd.read_csv(all_files[0], usecols=["Time"]).iloc[-1, 0]
    final_timestamp = datetime.strptime(final_timestamp_str.strip(), "%Y-%m-%d %H:%M:%S")
    
    # Extract only the base filenames (e.g., 'cat-1091162.csv') from the full paths
    all_files = [os.path.basename(f) for f in all_files]
    
    # Package the list of files, number of time steps, and final timestamp into a run configuration
    run_sets = [
        {
            'qlat_files': all_files,
            'nts': nts,
            'final_timestamp': final_timestamp
        }
    ]
```

**Purpose:**
- Discover all catchment files matching the `cat-*` pattern
- Extract simulation end time from the first catchment file
- Create a run configuration with all catchment files for processing

## Technical Details

### File Naming Convention

Catchment files follow the naming pattern: `cat-<flowpath_id>.csv`

Example: `cat-1091162.csv` contains lateral flow data for flowpath ID 1091162.

### Catchment File Format

Each catchment file contains at minimum two columns:
- `Time`: Timestamp in format "YYYY-MM-DD HH:MM:SS"
- `Q_OUT`: Lateral flow output value

## Benefits

1. **Finer Spatial Resolution**: Routing is performed at the flowpath level rather than aggregated nexus level
2. **Improved Accuracy**: Each segment receives its specific lateral flow contribution
3. **Enhanced Data Assimilation**: Direct gage-to-flowpath mapping enables accurate streamflow data assimilation at the segment level

## Configuration

To enable flowpath-level lateral flow support in T-Route, you need to modify the configuration file to specify the catchment file pattern.

### Configuration File Changes

In your T-Route configuration YAML file, update the `forcing_parameters` section as follows:

```yaml
forcing_parameters:
    #----------
    qts_subdivisions: 12
    dt: 300 # [sec] 300 == 5 minutes
    qlat_input_folder: ./outputs/ngen/
    qlat_file_pattern_filter: "cat-*"
    #binary_nexus_file_folder: ./outputs/parquet/ # if nexus_file_pattern_filter="nex-*" and you want it to reformat them as parquet, you need this
    #coastal_boundary_input_file : channel_forcing/schout_1.nc
    nts: 315072.0 #288 for 1day
    max_loop_size: 315072.0 # [number of timesteps]
    # max_loop_size == nts so that output is single file
```

### Key Configuration Parameters

- **`qlat_file_pattern_filter`**: Set to `"cat-*"` to enable catchment file processing
  - This tells T-Route to look for files matching the pattern `cat-<flowpath_id>.csv`
  - Previously, this would be set to `"nex-*"` for nexus file processing


### Example Usage

For a typical NextGen in a Box setup:

1. Set up NextGen in a Box from the [MSTHydroLab repository](https://github.com/said-uzzaman/t-route-MST.git)
2. Run your hydrological model (CFE, NoahOWP, etc.) to generate catchment output files
4. Update your T-Route configuration file with `qlat_file_pattern_filter: "cat-*"`
5. Run T-Route with the updated configuration

The routing will now be performed at the flowpath level with properly scaled lateral flows.


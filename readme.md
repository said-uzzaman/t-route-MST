# T-Route Modifications: Flowpath-Level Routing and Data Assimilation

## T-Route (Tree-Based Channel Routing)
To know more about T-Route, please click this link: https://github.com/CIROH-UA/t-route/blob/ngiab/readme.md

## Overview

This document describes modifications made to T-Route for use within NextGen in a Box. The changes fall into two groups:

1. **Flowpath-Level Lateral Flow Support**: enabling T-Route to read catchment output files (`cat-*`) directly from hydrological models (e.g., CFE, NoahOWP), provide lateral flow to every flowpath, and route every segment in the basin.
2. **Data Assimilation (Nudging)**: deriving gage-to-flowpath mapping directly from the flowpaths data, enabling nudging in the diffusive wave (DW) kernel following the Muskingum-Cunge (MC) kernel, and instrumenting both kernels to store the assimilated flow at each gaged location for evaluation.

## Background

### Previous Implementation
- T-Route could only read nexus files; each nexus carried the cumulative lateral flow of one or more flowpaths.
- Routing was performed for only one of the flowpaths associated with each nexus.
- Streamflow gage mapping for DA used hydrosequence-based downstream filtering from the network data.
- Nudging-based DA was available in the MC kernel but was not instrumented to store the assimilated flow for evaluation.

### New Implementation
- T-Route reads individual catchment files (`cat-*`), each providing lateral flow for a specific flowpath.
- Routing is performed for every segment, with lateral flows scaled by catchment area and time units.
- Gage mapping is derived directly from the flowpaths data, enabling flowpath-level gage assignment for DA.
- We enabled nudging in the DW kernel (following MC), and both kernels store the assimilated flow (the predicted discharge at the gaged segment, recorded before it is replaced by the observation) for evaluation.
- The output save boundary uses a threshold-based scheme (`t_next_save`) that removes the floating-point truncation error in the `mod()`-based check, which occurs in any DW run, even in short simulations.

# Part 1: Flowpath-Level Lateral Flow Support

## 1. HYFeaturesNetwork.py

**File Path:** `src/troute-network/troute/HYFeaturesNetwork.py`

### Modification 1: Flowpath ID Extraction and Unit Factor Calculation

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

### Modification 2: Catchment File Processing

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

### Modification 3: Unit Scaling Function

**Function:** `scale_lateral_flows()`

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

## 2. AbstractNetwork.py

**File Path:** `src/troute-network/troute/AbstractNetwork.py`

### Modification: Catchment File Discovery and Configuration

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

---

# Part 2: Data Assimilation (Nudging)

This part describes the data-assimilation modifications. We perform streamflow data assimilation (DA) by nudging: at each time step, if a segment contains a gage and a valid USGS observation is available, we replace the modeled discharge in that segment with the observation. The modifications (1) derive the gage-to-flowpath mapping directly from the flowpaths data, (2) instrument the MC kernel to store the assimilated flow before it is replaced, and (3) enable nudging in the DW kernel, following MC, together with the same storage of the assimilated flow and a numerically stable save boundary.

At a gaged segment, the routed discharge at time *t* is the predicted discharge of the DA run, which already carries the corrections applied at the preceding time steps. For DA evaluation, we store this predicted discharge before it is replaced with the observed value; we refer to this stored series as the assimilated flow. Our primary motivation for storing the assimilated flow is to compare how the nudging correction propagates under the two routing formulations.

## 2.1 Gage Mapping for Data Assimilation

**File Path:** `src/troute-network/troute/HYFeaturesNetwork.py`

**Function:** `preprocess_data_assimilation()`

Modified to derive streamflow gage mapping directly from the flowpaths dataframe instead of using hydrosequence-based filtering from the network dataframe.

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

## 2.2 Storing the Assimilated Flow for MC

**File Path:** `src/troute-routing/troute/routing/fast_reach/mc_reach.pyx`

**Location:** Inside the compute-network routing loop, in the branch that handles reaches carrying a streamflow gage (`reach_has_gage[i] > -1`), immediately before the `simple_da()` nudging call.

The MC kernel performs nudging-based DA through `simple_da()`. Before `simple_da()` replaces the modeled discharge with the observation, we store the predicted value at the gaged segment. As described in the Part 2 introduction, this stored assimilated flow is what we use to evaluate the scheme against the observations and against the open-loop flow. We read the value from `flowveldepth` at the gage segment position and append it to `MC_DA.txt`.

```python
if reach_has_gage[i] > -1:
    # We only enter this process for reaches where the
    # gage actually exists.
    # If assimilation is active for this reach, we touch the
    # exactly one gage which is relevant for the reach ...
    gage_i = reach_has_gage[i]
    usgs_position_i = usgs_positions[gage_i]

    # Store the predicted (assimilated) flow BEFORE nudging replaces it
    with open("MC_DA.txt", "a") as f:   # "a" = append mode
        f.write(f"timestep={timestep}, usgs_position_i={usgs_position_i}, "
                f"gage={gage_i}, flow={flowveldepth[usgs_position_i, timestep, 0]}\n")

    da_buf = simple_da(
        timestep,
        routing_period,
        da_decay_coefficient,
        gage_maxtimestep,
        NAN if timestep >= gage_maxtimestep else usgs_values[gage_i, timestep],
        flowveldepth[usgs_position_i, timestep, 0],
        lastobs_times[gage_i],
        lastobs_values[gage_i],
        gage_i == da_check_gage,
    )
```

**Purpose:**
- Store the assimilated flow — the predicted MC discharge at each gaged segment — before `simple_da()` replaces it with the observation
- Capture the value at the exact `usgs_position_i` fed into `simple_da()`, so the stored series is directly comparable to the observation and to the open-loop flow
- Enable evaluation of the nudging effect on MC routing at the gaged segment

## 2.3 Nudging, Assimilated-Flow Storage, and Stable Save Boundary for DW

**File Path:** `src/kernel/diffusive/diffusive.f90`

We enabled nudging in the diffusive wave kernel, following the MC formulation, and instrumented it to store the assimilated flow in the same way. A companion numerical correction replaces the `mod()`-based save-boundary check with a threshold, so that the stored series and the output remain aligned with the intended recording interval.

### 2.3.1 Nudging and Assimilated-Flow Storage

**Subroutine:** `mesh_diffusive_forward()`

For a reach that carries USGS streamflow data (`usgs_da_reach(j) /= 0`), we nudge the discharge at the reach bottom node toward the observation interpolated to the current time. Before we replace `qp(ncomp, j)`, we store the predicted value (the assimilated flow) and write it to `DW_DA.txt` at save boundaries only, so that the stored series is aligned with the model output recording interval rather than the internal (adaptive) time step. Unlike the MC scheme, we apply no temporal decay: when no valid observation is available (missing or flagged, `<= -4443.999`), we keep the modeled value.

```fortran
! when a reach has usgs streamflow data at its location, apply DA
if (usgs_da_reach(j) /= 0) then

  ! store the predicted value at the reach bottom node (assimilated flow) before replacement
  q_before_DA = eei(ncomp) * qp_ghost + ffi(ncomp)

  ! initialize the DA save threshold on the first call
  if (t_next_save_da < 0.0) then
     t_next_save_da = t0 * 60.0 + saveInterval / 60.0
  end if

  ! open the log file once, at the first boundary write
  if (.not. preDA_file_opened) then
     open(unit=preDA_unit, file=preDA_fname, status='replace', action='write')
     write(preDA_unit,'(A)') 'timestep reach q'
     preDA_file_opened = .true.
  end if

  ! write the assimilated flow at save boundaries only (threshold-based, see Section 2.3.2)
  if ( (t + dtini/60.d0) >= t_next_save_da - TOLERANCE &
       .or. (t + dtini/60.d0 >= tfin*60.d0 - TOLERANCE) ) then
     it_da_safe = nint( (t + dtini/60.d0 - t0*60.d0)*60.d0 / saveInterval )
     write(preDA_unit,'(I12,1X,I8,1X,ES24.16)') it_da_safe, j, q_before_DA
     t_next_save_da = t_next_save_da + saveInterval / 60.0
  end if

  ! nudging: interpolate the USGS observation to the current time and replace the bottom-node value
  allocate(varr_da(nts_da))
  do n = 1, nts_da
      varr_da(n) = usgs_da(n, j)
  end do
  qp(ncomp,j) = intp_y(nts_da, tarr_da, varr_da, t + dtini/60.)
  flag_da = 1

  ! keep the modeled value when the observation is missing/poor quality
  irow = locate(tarr_da, t + dtini/60.)
  if (irow == nts_da) then
    irow = irow-1
  endif
  if ((varr_da(irow) <= -4443.999).or.(varr_da(irow+1) <= -4443.999)) then
    qp(ncomp,j)  = eei(ncomp) * qp_ghost + ffi(ncomp)
    flag_da = 0
  endif
  deallocate(varr_da)
else

  ! reach has no gage: no assimilation
  qp(ncomp,j)  = eei(ncomp) * qp_ghost + ffi(ncomp)
  flag_da = 0
endif
```

**Purpose:**
- Enable nudging-based streamflow DA in the DW kernel, following the MC scheme
- Store the assimilated flow — the predicted DW discharge at the gaged reach bottom node — before nudging replaces `qp(ncomp, j)`, and write one record per gaged reach per save boundary
- Keep the modeled value (the ghost-point solution) when the observation is unavailable or flagged, so that routing continues

### 2.3.2 Threshold-Based Save Boundary (`t_next_save`)

**Subroutines:** `diffnw()` and `mesh_diffusive_forward()`

**Problem:** Under adaptive time stepping, the accumulated simulation time `t` rarely coincides exactly with a multiple of the save interval. A `mod()`-based check (testing whether `t` is an integer multiple of the save interval) therefore suffers from floating-point truncation error and misses or shifts a save boundary. This occurs in any DW run, even in short simulations, and appears as a timing offset in the stored series.

**Fix:** The modulo test is replaced by an explicit running threshold. A variable (`t_next_save`, with its DA-logging counterpart `t_next_save_da`) is advanced by exactly `saveInterval / 60.0` after each write, and the boundary test uses a `TOLERANCE` guard:

```fortran
! initialize once
t_next_save = t0 * 60.0 + saveInterval / 60.0

! ... inside the time loop ...
if ( t >= t_next_save - TOLERANCE .or. t >= tfin * 60.0 - TOLERANCE ) then
    ! ... write results to output arrays ...
    t_next_save = t_next_save + saveInterval / 60.0
end if
```

Because we obtain the next boundary by repeated addition from a fixed origin rather than recomputing it from `t` through `mod()`, the comparison remains stable throughout the simulation. A companion diagnostic reports any skipped boundary:

```fortran
if ( t >= t_next_save + saveInterval / 60.0 - TOLERANCE ) then
    print*, 'WARNING: skipped save boundary! t=', t, 't_next_save=', t_next_save
end if
```

**Purpose:**
- Eliminate the floating-point truncation error in the save-boundary check, which occurs in any DW run, even in short simulations
- Keep the recorded output steps, and the stored assimilated flow, aligned with the intended recording interval
- Prevent one-step timing offsets in the DA-nudged diffusive hydrographs

---

## Technical Details

### File Naming Convention

Catchment files follow the naming pattern: `cat-<flowpath_id>.csv`

Example: `cat-1091162.csv` contains lateral flow data for flowpath ID 1091162.

### Catchment File Format

Each catchment file contains at minimum two columns:
- `Time`: Timestamp in format "YYYY-MM-DD HH:MM:SS"
- `Q_OUT`: Lateral flow output value

## Benefits

1. **Finer Spatial Resolution**: Routing is performed at the flowpath level rather than aggregated nexus level.
2. **Improved Accuracy**: Each segment receives its specific lateral flow contribution.
3. **Consistent Data Assimilation**: Direct gage-to-flowpath mapping enables accurate streamflow DA at the segment level.
4. **Cross-Kernel Nudging**: Nudging is now available in both the MC and DW kernels using a consistent approach.
5. **Evaluable Assimilation**: Storing the assimilated flow in both kernels lets us evaluate the nudging effect against observations and against the open-loop flow, and examine the upstream versus downstream propagation of corrections in DW relative to MC.
6. **Numerical Robustness**: The threshold-based save boundary removes the floating-point truncation error in the `mod()`-based check, which affects any DW run, keeping the output and the stored series correctly timed.

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
3. Update your T-Route configuration file with `qlat_file_pattern_filter: "cat-*"`
4. Run T-Route with the updated configuration

The routing will now be performed at the flowpath level with properly scaled lateral flows.

## Author
**Md Saiduzzaman**  
Graduate Research Assistant, Department of Civil, Architectural and Environmental Engineering  
Missouri University of Science and Technology, Rolla, MO, USA   
Email: [msmg8@mst.edu](mailto:msmg8@mst.edu)

**Dr. Bong-Chul Seo**  
Assistant Professor, Department of Civil, Architectural and Environmental Engineering  
Missouri University of Science and Technology, Rolla, MO, USA  
Contact: [bongchul.seo@mst.edu](mailto:bongchul.seo@mst.edu)

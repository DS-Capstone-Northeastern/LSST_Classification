import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def filter_and_load_chunks(filepath, valid_ids_df, chunksize=500000):
    """
    Safely loads 22GB CSVs by immediately dropping unneeded object_ids.
    valid_ids_df should be the loaded train_ids.csv or val_ids.csv.
    """
    valid_ids = set(valid_ids_df['object_id'])
    chunk_list = []
    
    # Process the file in chunks to prevent RAM exhaustion
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        filtered_chunk = chunk[chunk['object_id'].isin(valid_ids)]
        chunk_list.append(filtered_chunk)
        
    return pd.concat(chunk_list, ignore_index=True)

def engineer_temporal_features(lightcurve_df):
    """
    Calculates the continuous time gap (delta_t) to solve observation intermittency.
    """
    # Sort chronologically per object
    df = lightcurve_df.sort_values(['object_id', 'mjd'])
    
    # Calculate time since first observation for the sequence (t=0)
    df['time_since_start'] = df.groupby('object_id')['mjd'].transform(lambda x: x - x.min())
    
    # Calculate delta_t between consecutive observations
    df['delta_t'] = df.groupby('object_id')['mjd'].diff().fillna(0.0)
    
    return df

def scale_features(lightcurve_df, metadata_df):
    """
    Applies Robust Scaling to flux to prevent exploding gradients from supernova spikes.
    """
    # RobustScaler uses median and IQR, making it immune to massive anomaly spikes
    flux_scaler = RobustScaler()
    
    # Scale continuous dynamic features
    lightcurve_df[['flux', 'flux_err']] = flux_scaler.fit_transform(lightcurve_df[['flux', 'flux_err']])
    
    # Scale continuous static features (redshift, distance modulus, milky way extinction)
    static_cols = ['hostgal_photoz', 'hostgal_photoz_err', 'mwebv']
    static_scaler = RobustScaler()
    metadata_df[static_cols] = static_scaler.fit_transform(metadata_df[static_cols])
    
    return lightcurve_df, metadata_df, flux_scaler

def build_processed_dataset(lightcurve_path, metadata_path, valid_ids_path):
    """
    End-to-end execution combining the above functions.
    """
    valid_ids = pd.read_csv(valid_ids_path)
    
    # 1. Filtered Loading
    lc_df = filter_and_load_chunks(lightcurve_path, valid_ids)
    meta_df = pd.read_csv(metadata_path)
    meta_df = meta_df[meta_df['object_id'].isin(valid_ids['object_id'])]
    
    # 2. Feature Engineering
    lc_df = engineer_temporal_features(lc_df)
    lc_df, meta_df, _ = scale_features(lc_df, meta_df)
    
    # 3. Merge static into dynamic for sequence building
    final_df = lc_df.merge(meta_df, on='object_id', how='left')
    return final_df
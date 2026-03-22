import torch
from torch.utils.data import Dataset
import numpy as np

class PLAsTiCCTFTDataset(Dataset):
    def __init__(self, processed_df, max_seq_len=350, pad_value=0.0):
        """
        Expects the merged dataframe from preprocessing.py.
        Groups by object_id to build [seq_len, num_features] matrices.
        """
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        
        # Group data by object_id
        self.grouped = processed_df.groupby('object_id')
        self.object_ids = list(self.grouped.groups.keys())
        
        # Define feature columns
        self.static_cols = ['hostgal_photoz', 'hostgal_photoz_err', 'mwebv']
        self.dynamic_cont_cols = ['flux', 'flux_err', 'delta_t', 'time_since_start']
        self.dynamic_cat_cols = ['passband'] # Integers 0-5
        self.target_col = 'true_target'
        
        # Map class IDs (e.g., 42, 64, 90) to contiguous indices (0, 1, 2...) for CrossEntropyLoss
        unique_targets = processed_df[self.target_col].unique()
        self.target_map = {val: idx for idx, val in enumerate(sorted(unique_targets))}
        self.num_classes = len(self.target_map)

    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, idx):
        obj_id = self.object_ids[idx]
        group = self.grouped.get_group(obj_id)
        
        # 1. Extract Static Features (constant across the sequence)
        static_feats = group[self.static_cols].iloc[0].values.astype(np.float32)
        target_val = group[self.target_col].iloc[0]
        mapped_target = self.target_map[target_val]
        
        # 2. Extract Dynamic Features
        dyn_cont = group[self.dynamic_cont_cols].values.astype(np.float32)
        dyn_cat = group[self.dynamic_cat_cols].values.astype(np.int64)
        
        seq_len = len(dyn_cont)
        
        # Truncate if sequence is unusually long
        if seq_len > self.max_seq_len:
            dyn_cont = dyn_cont[:self.max_seq_len]
            dyn_cat = dyn_cat[:self.max_seq_len]
            seq_len = self.max_seq_len
            
        # 3. Apply Padding
        pad_len = self.max_seq_len - seq_len
        
        padded_cont = np.pad(dyn_cont, ((0, pad_len), (0, 0)), 
                             mode='constant', constant_values=self.pad_value)
        # Pad categorical with 0 (will be ignored by mask anyway)
        padded_cat = np.pad(dyn_cat, ((0, pad_len), (0, 0)), 
                            mode='constant', constant_values=0)
        
        # 4. Generate Boolean Attention Mask (True for padding, False for actual data in standard PyTorch)
        # However, for manual masking, 1 for real, 0 for pad is often easier. Let's do 1=real, 0=pad.
        attention_mask = np.zeros(self.max_seq_len, dtype=np.bool_)
        attention_mask[:seq_len] = True
        
        return {
            'static': torch.tensor(static_feats),
            'dyn_cont': torch.tensor(padded_cont),
            'dyn_cat': torch.tensor(padded_cat).squeeze(-1),
            'mask': torch.tensor(attention_mask),
            'target': torch.tensor(mapped_target, dtype=torch.long)
        }
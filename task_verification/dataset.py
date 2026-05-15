##Task Verification dataset with data augmentation

import numpy as np
import json
import torch
from torch.utils.data import Dataset

class TaskVerificationDataset(Dataset):
    def __init__(self, npz_path, annotations_path, video_ids, split='train', max_seq_len=100):
        self.features = np.load(npz_path)
        self.split = split
        self.max_seq_len = max_seq_len
        
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
            
        self.video_list = [vid for vid in video_ids if vid in self.features]

    def __len__(self):
        return len(self.video_list)

    def apply_augmentation(self, feat):
        # gaussian noise
        if np.random.rand() > 0.4:
            noise = np.random.normal(0, 0.05, feat.shape)
            feat = feat + noise

        # step dropout
        if np.random.rand() > 0.4:
            # mask about 10% of the steps
            mask = np.random.rand(feat.shape[0], 1) > 0.2
            feat = feat * mask

        # jittering (Feature Scaling)
        if np.random.rand() > 0.4:
            scale_factor = np.random.uniform(0.85, 1.15)
            feat = feat * scale_factor
            
        return feat

    def __getitem__(self, idx):
        video_id = self.video_list[idx]
        feat = self.features[video_id].astype(np.float32)
        
        # Apply augmentation only during training
        if self.split == 'train':
            feat = self.apply_augmentation(feat)
        
        seq_len = feat.shape[0]
        has_error = any(step.get('has_errors', False) for step in self.annotations[video_id].get('steps', []))
        label = 1 if has_error else 0
        
        # Padding logic
        padded_feat = np.zeros((self.max_seq_len, 768), dtype=np.float32)
        actual_len = min(seq_len, self.max_seq_len)
        padded_feat[:actual_len, :] = feat[:actual_len, :]
        
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:actual_len] = 1.0
        
        return {
            'features': torch.tensor(padded_feat),
            'label': torch.tensor(label, dtype=torch.long),
            'mask': torch.tensor(mask),
            'video_id': video_id
        }


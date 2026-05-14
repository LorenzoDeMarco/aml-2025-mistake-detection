import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader

class TaskVerificationDataset(Dataset):
    def __init__(self, npz_path, annotations_path, video_ids, max_seq_len=100):
        """
        Args:
            npz_path: Percorso al file .npz contenente gli embedding [N, 768]
            annotations_path: Percorso al file complete_step_annotations.json
            video_ids: Lista di video_id estratti dallo split ufficiale per questo set
            max_seq_len: Lunghezza massima della sequenza per il Transformer (Padding)
        """
        self.features = np.load(npz_path)
        self.max_seq_len = max_seq_len
        
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
            
        #filter to ensure we only keep videos that are in the features
        self.video_list = [vid for vid in video_ids if vid in self.features]
        
    def __len__(self):
        return len(self.video_list)
        
    def __getitem__(self, idx):
        video_id = self.video_list[idx]
        
        # 1. we load the features for the given video_id shape [N, 768]
        feat = self.features[video_id]  
        seq_len = feat.shape[0]
        
        # 2. we extract the global label from the annotations (1 if has_errors, 0 otherwise)
        has_error = any(step.get('has_errors', False) for step in self.annotations[video_id].get('steps', []))
        label = 1 if has_error else 0
        
        # 3. padding to ensure all sequences have the same length (max_seq_len)
        padded_feat = np.zeros((self.max_seq_len, 768), dtype=np.float32)
        
        if seq_len <= self.max_seq_len:
            padded_feat[:seq_len, :] = feat
            actual_len = seq_len
        else:
            padded_feat[:, :] = feat[:self.max_seq_len, :]
            actual_len = self.max_seq_len
            
        # 4.mask to indicate which parts of the sequence are valid (1 for valid, 0 for padded)
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:actual_len] = 1.0
        
        return {
            'features': torch.tensor(padded_feat, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'video_id': video_id
        }


def get_data_loaders(npz_path, annotations_path, split_json_path, batch_size=16, max_seq_len=100):
    with open(split_json_path, 'r') as f:
        split_data = json.load(f)
        
    train_vids = split_data['train']
    val_vids = split_data['val']
    test_vids = split_data['test']
    
    print("=== Official Dataset Split Loaded ===")
    print(f"Train videos: {len(train_vids)}")
    print(f"Val videos:   {len(val_vids)}")
    print(f"Test videos:  {len(test_vids)}")
    
    train_dataset = TaskVerificationDataset(npz_path, annotations_path, train_vids, max_seq_len)
    val_dataset = TaskVerificationDataset(npz_path, annotations_path, val_vids, max_seq_len)
    test_dataset = TaskVerificationDataset(npz_path, annotations_path, test_vids, max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

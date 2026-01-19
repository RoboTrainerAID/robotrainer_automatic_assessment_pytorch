import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PytorchDataset(Dataset):
    def __init__(self, X: torch.tensor, y: torch.tensor):
        """
        X: torch.tensor (N, Time, Features)
        y: torch.tensor (N, Targets)
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloader(X: torch.tensor, y: torch.tensor, batch_size, shuffle=True):
    dataset = PytorchDataset(X, y)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        pin_memory=True # Essential for fast GPU transfer
    )

def get_root_groups(users):
    """
    Maps augmented user IDs back to original user IDs. 
    Assumes augmentation user_id * 100 + i and original IDs < 100.
    """
    us = np.array(users)
    return np.array([u if u < 100 else u // 100 for u in us])



class AugmentedLOGO:
    """
    Custom splitter for Augmented Data.
    
    Behavior:
    1. Iterates over each 'real' user (ID < 100) as the Test Set.
    2. Drops all augmented versions (clones) of the current Test User from the Train Set.
       (Ensures we don't train on augmented versions of the subject we are testing on).
    3. Train set includes all other users AND their augmented versions.
    
    Args:
        include_augmented_in_test (bool): If True, the test set includes the original user 
            AND their augmented clones. If False, only the original user is tested.
            In both cases, clones are excluded from the training set.
    """
    def __init__(self, include_augmented_in_test: bool = False):
        self.include_augmented_in_test = include_augmented_in_test

    def split(self, groups):
        groups = np.array(groups)
        root_groups = get_root_groups(groups)
        unique_roots = np.unique(root_groups)
        
        for root in unique_roots:
            # Identification of relevant records for this root (original + augmented)
            # We want to exclude ALL of them from training if we are testing on the root
            root_family_mask = (root_groups == root)
            
            if self.include_augmented_in_test:
                test_mask = root_family_mask
            else:
                # Test: Specifically records that match the ROOT ID exactly (original data only)
                test_mask = (groups == root)
            
            if np.sum(test_mask) == 0:
                continue

            
            # Train: Everything that is NOT part of the current root's family
            train_mask = ~root_family_mask
            
            yield np.where(train_mask)[0], np.where(test_mask)[0]
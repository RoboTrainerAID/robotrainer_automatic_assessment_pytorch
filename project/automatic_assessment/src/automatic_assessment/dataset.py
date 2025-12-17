import pandas as pd

from automatic_assessment.data_preprocessing import Preprocessor

class Dataset:
    def __init__(self, sampling_frequency, path, recreate):
        if recreate:
            print(f"Creating dataset at {path} (Sampling: {sampling_frequency} Hz)...")
            prep = Preprocessor(sampling_frequency, path)
            self.data = prep._get_dataset()
        else:
            print(f"Loading existing dataset from {path}...")
            self.data = pd.read_csv(path)

        self.length = len(self.data)

        self._validate_data()

    def _validate_data(self):
        """Basic validation after loading."""
        num_features = len([c for c in self.df.columns if c not in self.TARGET_COLS])
        num_targets = len(self.TARGET_COLS)
        print(f"Dataset Loaded. Shape: {self.df.shape}. Users: {self.df['user'].nunique()}. Features: {num_features}. Targets: {num_targets}")
    
    
class DatasetConv1s(Dataset):
    def __init__(self, recreate: bool = False):
        PATH = "/data/dataset_conv@1s.csv"
        super().__init__(sampling_frequency=1, path=PATH, recreate=recreate)

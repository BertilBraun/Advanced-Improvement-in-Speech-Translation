import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def iterate_over_dataset(dataset: Dataset, desc="", start=0, end=None):
    end = len(dataset) if end is None else end
    for i in tqdm(range(start, end), desc=desc):
        try:
            data = dataset[i]
        except Exception as e:
            print(f"Error loading dataset at {desc} {i}: {e}")
            continue
        yield data


class ConcatenatedDatasets(Dataset):
    """Concatenate multiple datasets together."""
    
    def __init__(self, datasets: list[Dataset]) -> None:
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cum_lengths = np.cumsum(self.lengths)
        self.total_length = self.cum_lengths[-1]
        
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int):
        dataset_idx = np.searchsorted(self.cum_lengths, idx, side="right")
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cum_lengths[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
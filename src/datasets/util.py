import numpy as np
from typing import Iterator, TypeVar
from torch.utils.data import Dataset
from tqdm import tqdm

from src.logger_utils import get_logger

logger = get_logger("Dataset::Utils")


# Create a generic type variable
T = TypeVar('T')

def iterate_over_dataset(dataset: Dataset[T], desc: str = "", start: int = 0, end: int | None = None) -> Iterator[T]:
    end = len(dataset) if end is None else end
    for i in tqdm(range(start, end), desc=desc):
        try:
            yield dataset[i]
        except Exception as e:
            logger.error(f"Error loading dataset at {desc} {i}: {e}")


class ConcatenatedDatasets(Dataset[T]):
    """Concatenate multiple datasets together."""

    def __init__(self, datasets: list[Dataset[T]]) -> None:
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cum_lengths = np.cumsum(self.lengths)
        self.total_length = self.cum_lengths[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> T:
        dataset_idx = np.searchsorted(self.cum_lengths, idx, side="right")
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cum_lengths[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

import numpy as np

from tqdm import tqdm
from typing import Callable, Iterator, TypeVar
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.logger_utils import get_logger

logger = get_logger('Dataset::Utils')


# Create a generic type variable
T = TypeVar('T')
R = TypeVar('R')


def iterate_over_dataset_slow_sequential(dataset: Dataset[T], desc: str = '') -> Iterator[T]:
    for i in tqdm(range(len(dataset)), desc=desc):
        try:
            yield dataset[i]
        except Exception as e:
            logger.error(f'Error loading dataset at {desc} {i}: {e}')


def iterate_over_dataset(dataset: Dataset[T], desc: str = '', max_workers=10, batch_size=100) -> Iterator[T]:
    def process_element(el: T) -> T:
        return el

    for el in process_dataset_in_parallel(
        dataset, process_element, desc=desc, max_workers=max_workers, batch_size=batch_size
    ):
        if el is not None:
            yield el


def process_dataset_in_parallel(
    dataset: Dataset[T], func: Callable[[T], R], desc: str = '', max_workers=10, batch_size=100
) -> Iterator[R | None]:
    def process_at_index(index: int) -> R | None:
        try:
            return func(dataset[index])
        except Exception as e:
            logger.error(f'Error processing dataset at index {index}: {e}')
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        total_files = len(dataset)
        for batch_start in tqdm(range(0, total_files, batch_size), desc=desc):
            future_to_index = {
                executor.submit(process_at_index, i): i
                for i in range(batch_start, min(batch_start + batch_size, total_files))
            }
            for future in as_completed(future_to_index):
                yield future.result()

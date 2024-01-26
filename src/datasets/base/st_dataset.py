from typing import Optional

from torch.utils.data import Dataset

from src.logger_utils import get_logger

logger = get_logger("Dataset::ST_Dataset")

DataSample = tuple[str, str, Optional[str], str, str] # waveform_path, sentence, translation, speaker_id, sample_id

class STDataset(Dataset[DataSample]):
    def __init__(self) -> None:
        self._data = self._load_data()

    def _load_data(self) -> list[DataSample]:
        raise NotImplementedError

    def __getitem__(self, n: int) -> DataSample:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(waveform_path, sentence, translation, speaker_id, sample_id)``
        """
        return self._data[n]

    def __len__(self) -> int:
        return len(self._data)


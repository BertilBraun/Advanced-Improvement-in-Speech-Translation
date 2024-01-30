from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

from src.logger_utils import get_logger

logger = get_logger("Dataset::ST_Dataset")

DataSample = tuple[Path, str, Optional[str], str, str] # waveform_path, sentence, translation, speaker_id, sample_id

class STDataset(Dataset[DataSample]):
    def __init__(self, split: str) -> None:
        self.split = split
        self._data = self._load_data()
        self._data = self.__filter_data()

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

    def __filter_data(self) -> list[DataSample]:
        filtered_data: list[DataSample] = []
        
        for path, sentence, translation, speaker_id, sample_id in self._data:
            if not path.is_file():
                logger.warning(f"Missing audio file for {speaker_id}-{sample_id}!")
                continue
            
            # if sentence is not a string, skip this sample
            if not isinstance(sentence, str) or \
                (not isinstance(translation, str) and translation is not None) or \
                not isinstance(speaker_id, str) or \
                not isinstance(sample_id, str):
                logger.warning(f"Skipping {path} because data type is wrong! (sentence: {sentence}) (translation: {translation}) (speaker_id: {speaker_id}) (sample_id: {sample_id})")
                continue
            
            filtered_data.append((path, sentence, translation, speaker_id, sample_id))
            
        return filtered_data

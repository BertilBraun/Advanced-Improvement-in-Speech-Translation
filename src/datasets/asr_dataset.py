from typing import Optional

import librosa
import torch
from src.datasets.util import iterate_over_dataset
from torch.utils.data import Dataset

DataSample = tuple[str, str, Optional[str], str, str]
ASRSample = tuple[torch.Tensor, int, str, Optional[str], str, str] # waveform, sample_rate, sentence, translation, speaker_id, sample_id

class ASRDataset(Dataset[ASRSample]):
    """Create a Dataset for ASR."""

    def __init__(self, datasource: Dataset[DataSample]) -> None:
        self.datasource = datasource

    def __getitem__(self, n: int) -> ASRSample:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(waveform, sample_rate, sentence, translation, speaker_id, sample_id)``
        """
        path, sentence, translation, speaker_id, sample_id = self.datasource[n]
        waveform, sample_rate = librosa.load(path, sr=None, mono=False)
        waveform = torch.tensor(waveform)[None, :]
        sample_rate = int(round(sample_rate, ndigits=0))
        return waveform, sample_rate, sentence, translation, speaker_id, sample_id


if __name__ == "__main__":
    from src.paths import COVOST_ROOT
    from src.datasets.covost import CoVoST

    datasource = CoVoST(COVOST_ROOT, "train", "en", "de")
    dataset_with_audio = ASRDataset(datasource)

    for waveform, sample_rate, sentence, translation, speaker_id, sample_id in iterate_over_dataset(dataset_with_audio):
        pass

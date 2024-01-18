import librosa
import torch

from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Optional


class ASRDataset(Dataset):
    """Create a Dataset for ASR."""

    def __init__(self, datasource: Dataset) -> None:
        self.datasource = datasource

    def __getitem__(
        self, n: int
    ) -> tuple[torch.Tensor, int, str, str, Optional[str], str, str]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(waveform, sample_rate, sentence, translation, speaker_id,
            sample_id)``
        """
        path, sample_rate, sentence, translation, speaker_id, sample_id = self.datasource[n]
        waveform, sample_rate = librosa.load(path, sr=None, mono=False)
        waveform = torch.tensor(waveform)[None, :]
        return waveform, sample_rate, sentence, translation, speaker_id, sample_id

if __name__ == "__main__":
    from src.datasets.covost import CoVoST

    COVOST_ROOT = "/pfs/work7/workspace/scratch/ubppd-ASR/covost"

    datasource = CoVoST(COVOST_ROOT, "train", "en", "de")
    dataset_with_audio = ASRDataset(datasource)
    
    for waveform, sample_rate, sentence, translation, speaker_id, sample_id in tqdm(dataset_with_audio):
        pass
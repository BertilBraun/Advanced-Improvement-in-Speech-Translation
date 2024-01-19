import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets.util import iterate_over_dataset
from torch.utils.data import Dataset

from utils import get_logger

logger = get_logger("Dataset::Covost2")


class CoVoST(Dataset):
    """Create a Dataset for CoVoST (https://github.com/facebookresearch/covost).
    Args:
        root (str): root path to the dataset and generated manifests/features
        source_language (str): source (audio) language
        target_language (str): target (text) language
    """

    SPLITS = ["train", "dev", "test"]

    def __init__(
        self,
        root: str,
        split: str,
        source_language: str,
        target_language: str,
    ) -> None:
        self.root = Path(root)
        self.source_language = source_language
        self.target_language = target_language
        self.data = (
            pd.read_csv(
                self.root / "translations" / f"covost_v2.en_de.{split}.tsv",
                sep="\t",
                on_bad_lines="warn",
            )
            .to_dict(orient="index")
            .items()
        )

    def __getitem__(self, n: int) -> tuple[str, int, str, Optional[str], str, str]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(waveform_path, sample_rate, sentence, translation, speaker_id,
            sample_id)``
        """
        data = self.data[n]
        path = self.root / "corpus-16.1" / self.source_language / "clips" / data["path"]
        sentence = data["sentence"]
        translation = data["translation"]
        speaker_id = data["client_id"]
        sample_id = data["path"].replace(".mp3", "")
        return path, sample_rate, sentence, translation, speaker_id, sample_id

    def __len__(self) -> int:
        return len(self.data)


class CoVoSTWithText(Dataset):
    """Create a Dataset for CoVoST (https://github.com/facebookresearch/covost).
    Args:
    root (str): root path to the dataset and generated manifests/features
    source_language (str): source (audio) language
    target_language (str): target (text) language
    """

    def __init__(
        self,
        root: str,
        split: str,
        source_language: str,
        target_language: str,
    ) -> None:
        self.dataset = CoVoST(root, split, source_language, target_language)

    def __getitem__(self, n: int) -> tuple[str, str]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(sentence, translation)``
        """
        path, sample_rate, sentence, translation, speaker_id, sample_id = self.dataset[
            n
        ]
        return sentence, translation


if __name__ == "__main__":
    COVOST_ROOT = "/pfs/work7/workspace/scratch/ubppd-ASR/covost"
    SOURCE_LANG_ID = "en"
    TARGET_LANG_ID = "de"

    for split in CoVoST.SPLITS:
        logger.info(f"Fetching split {split}...")
        dataset = CoVoST(COVOST_ROOT, split, SOURCE_LANG_ID, TARGET_LANG_ID)

        for (
            path,
            sample_rate,
            sentence,
            translation,
            speaker_id,
            sample_id,
        ) in iterate_over_dataset(dataset):
            pass

        dataset_with_text = CoVoSTWithText(
            COVOST_ROOT, split, SOURCE_LANG_ID, TARGET_LANG_ID
        )

        for sentence, translation in iterate_over_dataset(dataset_with_text):
            pass

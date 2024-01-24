import pandas as pd

from pathlib import Path

from src.datasets.mt_dataset import MTDataset, TextSample
from src.datasets.st_dataset import STDataset, DataSample
from src.datasets.util import iterate_over_dataset
from src.logger_utils import get_logger
from src.paths import COVOST_ROOT

logger = get_logger("Dataset::Covost2")

class CoVoST(STDataset):
    """Create a Dataset for CoVoST (https://github.com/facebookresearch/covost).
    Args:
        root (Path): root path to the dataset and generated manifests/features
        source_language (str): source (audio) language
        target_language (str): target (text) language
    """

    SPLITS = ["train", "dev", "test"]

    def __init__(self, root: Path, split: str, source_language: str, target_language: str) -> None:
        self.root = root
        self.split = split
        self.source_language = source_language
        self.target_language = target_language
        super().__init__()
        
    def _load_data(self) -> list[DataSample]:
        data = []
        
        for element in pd.read_csv(
            self.root / "data" / "translations" / f"covost_v2.en_de.{self.split}.tsv",
            sep="\t",
            on_bad_lines="warn",
        ).to_dict(orient="index").values():
            path = self.root / "corpus-16.1" / self.source_language / "clips" / element["path"]
            sentence = element["sentence"]
            translation = element["translation"]
            speaker_id = element["client_id"]
            sample_id = element["path"].replace(".mp3", "")
            
            data.append((path, sentence, translation, speaker_id, sample_id))
            
        return data


class CoVoSTWithText(MTDataset):
    """Create a Dataset for CoVoST (https://github.com/facebookresearch/covost).
    Args:
    root (Path): root path to the dataset and generated manifests/features
    source_language (str): source (audio) language
    target_language (str): target (text) language
    """

    def __init__(self, root: Path, split: str, source_language: str, target_language: str) -> None:
        self.dataset = CoVoST(root, split, source_language, target_language)
        
    @property
    def split(self) -> str:
        return self.dataset.split
        
    def _load_data(self) -> list[TextSample]:
        data = []
        for path, sentence, translation, speaker_id, sample_id in iterate_over_dataset(self.dataset):
            data.append((sentence, translation))
        return data


if __name__ == "__main__":
    for split in CoVoST.SPLITS:
        logger.info(f"Fetching split {split}...")
        dataset = CoVoST(COVOST_ROOT, split, "en", "de")

        for (path, sentence, translation, speaker_id, sample_id) in iterate_over_dataset(dataset):
            pass

        dataset_with_text = CoVoSTWithText(COVOST_ROOT, split, "en", "de")

        for sentence, translation in iterate_over_dataset(dataset_with_text):
            pass

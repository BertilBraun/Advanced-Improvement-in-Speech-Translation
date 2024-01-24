from datasets import load_dataset

from src.datasets.mt_dataset import MTDataset, TextSample
from src.datasets.util import iterate_over_dataset
from src.logger_utils import get_logger

logger = get_logger("Dataset::WMT")

class WMT(MTDataset):
    """Create a Dataset for WMT.
    Args:
        wmts (list[int]): list of WMT versions to use
        source_language (str): source (text) language
        target_language (str): target (text) language
    """

    SPLITS = ["train", "dev", "test"]

    def __init__(self, wmts: list[int], split: str, source_language: str, target_language: str, max_size: int | None = None) -> None:
        self.wmts = wmts
        self.split = split
        self.source_language = source_language
        self.target_language = target_language
        self.max_size = max_size
        super().__init__()

    def _load_data(self) -> list[TextSample]:
        # WMT datasets, but generate a set from these to avoid duplicates

        language = f"{self.source_language}-{self.target_language}"
        s = set()

        for wmt in self.wmts:
            logger.info(f"Loading WMT{wmt} dataset...")
            dataset = load_dataset(
                f"wmt{wmt}", language, split=self.split, trust_remote_code=True
            )
            logger.info(f"Now processing WMT{wmt} dataset...")

            # extend the set with the new sentences
            s.update(
                (sentence["en"], sentence["de"])                             # type: ignore
                for sentence in dataset["translation"]                       # type: ignore
                if sentence["en"] is not None and sentence["de"] is not None # type: ignore
            )

            if self.max_size is not None and len(s) >= self.max_size:
                break

        # convert the set to a list
        s = list(s)
        if self.max_size is not None:
            s = s[: self.max_size]
        return s


if __name__ == "__main__":
    SOURCE_LANG_ID = "en"
    TARGET_LANG_ID = "de"
    WMTS = list(range(19, 14, -1))
    MAX_DATASET_SIZE = 10_000_000

    for split in WMT.SPLITS:
        logger.info(f"Fetching split {split}...")
        dataset = WMT(WMTS, split, SOURCE_LANG_ID, TARGET_LANG_ID, MAX_DATASET_SIZE)

        for en, de in iterate_over_dataset(dataset):
            pass

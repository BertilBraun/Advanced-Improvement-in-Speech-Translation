from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset

class WMT(Dataset):
    """Create a Dataset for WMT.
    Args:
        wmts (list[int]): list of WMT versions to use
        source_language (str): source (text) language
        target_language (str): target (text) language
    """

    SPLITS = ["train", "dev", "test"]

    def __init__(
        self,
        wmts: list[int],
        split: str,
        source_language: str,
        target_language: str,
        max_size: int = None,
    ) -> None:
        self.wmts = wmts
        self.split = split
        self.source_language = source_language
        self.target_language = target_language
        self.max_size = max_size
        self.data = self.load_datasets()
        

    def __getitem__(
        self, n: int
    ) -> tuple[str, str]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(sentence, translation)``
        """
        return self.data[n]

    def __len__(self) -> int:
        return len(self.data)
    
    def load_datasets(self) -> list[tuple[str, str]]:
        # WMT datasets, but generate a set from these to avoid duplicates

        language = f"{self.source_language}-{self.target_language}"
        s = set()    
        
        for wmt in self.wmts:
            print(f"Loading WMT{wmt} dataset...")
            dataset = load_dataset(f"wmt{wmt}", language, split=self.split, trust_remote_code=True)
            print(f"Now processing WMT{wmt} dataset...")
            
            # extend the set with the new sentences
            s.update(
                (sentence["en"], sentence["de"]) for sentence in dataset["translation"]
            )
            
            if self.max_size is not None and len(s) >= self.max_size:
                break

        # convert the set to a list
        s = list(s)
        if self.max_size is not None:
            s = s[:self.max_size]
        return s



if __name__ == "__main__":
    SOURCE_LANG_ID = "en"
    TARGET_LANG_ID = "de"
    WMTS = list(range(19, 14, -1))
    MAX_DATASET_SIZE = 10_000_000
    
    for split in WMT.SPLITS:
        print(f"Fetching split {split}...")
        dataset = WMT(WMTS, split, SOURCE_LANG_ID, TARGET_LANG_ID, MAX_DATASET_SIZE)

        for en, de in tqdm(dataset):
            pass
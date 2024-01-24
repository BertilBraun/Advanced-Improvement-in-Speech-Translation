from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

DO_FILTER_NON_ASCII = True
ALLOWED_NON_ASCII_CHARS = "–“’‘„”�…€—βüöäÜÖÄ"


def translation_pair_check(en: str, de: str) -> bool:
    # only keep sentences that are only ascii characters
    def ascii_check(s: str) -> bool:
        return s is not None and all(ord(c) < 256 or c in ALLOWED_NON_ASCII_CHARS for c in s)

    return not DO_FILTER_NON_ASCII or (ascii_check(en) and ascii_check(de))


def cleanup(src: str, tgt: str) -> tuple[str, str]:
    def clean(s: str) -> str:
        return s.replace("\n", " ").replace("\t", " ").strip()

    return clean(src), clean(tgt)


TextSample = tuple[str, str]
class MTDataset(Dataset[TextSample]):
    """Create a clean Dataset for MT."""

    def __init__(self) -> None:
        self.data = self.__load_datasets()

    def __getitem__(self, n: int) -> TextSample:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(sentence, translation)``
        """
        return self.data[n]

    def __len__(self) -> int:
        return len(self.data)

    def write_to_files(self, src_path: Path, tgt_path: Path, max_lines_to_write: int | None = None) -> None:
        with open(src_path, "w") as src_file, open(tgt_path, "w") as tgt_file:
            for i, (src, tgt) in tqdm(enumerate(self.data), desc="Writing to files"):
                if max_lines_to_write is not None and i >= max_lines_to_write:
                    break
                src_file.write(src + "\n")
                tgt_file.write(tgt + "\n")
                
    def _load_data(self) -> list[TextSample]:
        raise NotImplementedError()

    def __load_datasets(self) -> list[tuple[str, str]]:
        return [
            cleanup(src, tgt)
            for src, tgt in tqdm(self._load_data(), desc="Processing dataset")
            if translation_pair_check(src, tgt)
        ]

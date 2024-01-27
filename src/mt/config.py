from typing import Sequence
from src.datasets.base.mt_dataset import MTDataset
from src.logger_utils import get_logger
from src.mt.bpe import BPE
from src.paths import *

logger = get_logger("MT::Config")

def process_mt_dataset_to_spm_encoding(datasets: Sequence[MTDataset], output_root: Path, spm_model: Path | None = None) -> None:   
    for dataset in datasets:
        split = dataset.split
        
        logger.info(f"Writing MT Dataset {split} to disk...")
        src_path = output_root / "data" / f"{split}.en"
        tgt_path = output_root / "data" / f"{split}.de"
        dataset.write_to_files(src_path, tgt_path, max_lines_to_write=20_000_000)

        logger.info("Loading MT BPE...")        
        bpe = BPE(
            retrain_spm=False,
            model_file=spm_model or output_root / "spm_unigram10000.model",
            train_files=[src_path.as_posix(), tgt_path.as_posix()],
        )
        
        logger.info("Encoding MT Dataset with sentencepiece...")
        logger.info(f"Encoding split {split}...")
        
        bpe.encode_file(
            output_root / "data" / f"{split}.en",
            output_root / "spm" / f"{split}.en",
        )
        
        bpe.encode_file(
            output_root / "data" / f"{split}.de",
            output_root / "spm" / f"{split}.de",
        )
        
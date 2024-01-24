import os

from src.asr.config import create_asr_configs
from src.asr.mel_encoding import process_dataset_to_mel_spectrogram
from src.datasets.asr_dataset import ASRDataset
from src.datasets.covost import CoVoST, CoVoSTWithText
from src.logger_utils import get_logger
from src.mt.bpe import BPE
from src.paths import *

logger = get_logger("Dataset::Prepare Datasets")

def can_skip_dataset_prep() -> bool:
    # if asr config file exists and mt spm data exists, we can skip
    config_file_exists = (ASR_COVOST_ROOT / "config.yaml").is_file()
    mt_spm_data_exists = all(
        (MT_COVOST_SPM_ENCODING_ROOT / f"{split}.en").is_file()
        for split in CoVoST.SPLITS
    )
    return config_file_exists and mt_spm_data_exists

if __name__ == "__main__":
    if can_skip_dataset_prep():
        logger.info("Skipping dataset preparation, config file and MT spm data already exists")
        exit(0)
    
    logger.info("Preparing CoVoST...")
    covost_datasets = [CoVoST(COVOST_ROOT, split, "en", "de") for split in CoVoST.SPLITS]
    
    logger.info("Processing CoVoST to mel spectrograms...")
    for dataset in covost_datasets:
        process_dataset_to_mel_spectrogram(ASRDataset(dataset), ASR_COVOST_MEL_ENCODING_ROOT)

    # copy spm model ASR_SPM_MODEL to ASR_COVOST_ROOT
    os.system(f"cp {ASR_SPM_MODEL} {ASR_COVOST_ROOT}")

    logger.info("Creating ASR configs for CoVoST...")
    create_asr_configs(
        covost_datasets,
        CoVoST.SPLITS,
        ASR_COVOST_ROOT,
        ASR_COVOST_MEL_ENCODING_ROOT,
        ASR_COVOST_MEL_ENCODING_ZIP,
        spm_filename=ASR_SPM_MODEL.name,
    )    
    
    logger.info("Preparing MT CoVoST...")
    mt_datasets = [CoVoSTWithText(COVOST_ROOT, split, "en", "de") for split in CoVoST.SPLITS]
    
    logger.info("Writing MT CoVoST to disk...")
    for dataset in mt_datasets:
        src_path = MT_COVOST_DATA_ROOT / f"{dataset.split}.en"
        tgt_path = MT_COVOST_DATA_ROOT / f"{dataset.split}.de"
        dataset.write_to_files(src_path, tgt_path, max_lines_to_write=20_000_000)

    logger.info("Loading MT BPE...")        
    bpe = BPE(
        retrain_spm=False,
        model_file=MT_SPM_MODEL,
    )
    
    logger.info("Encoding MT CoVoST with sentencepiece...")
    for split in CoVoST.SPLITS:
        logger.info(f"Encoding split {split}...")
        
        bpe.encode_file(
            MT_COVOST_DATA_ROOT / f"{split}.en",
            MT_COVOST_SPM_ENCODING_ROOT / f"{split}.en",
        )
        
        bpe.encode_file(
            MT_COVOST_DATA_ROOT / f"{split}.de",
            MT_COVOST_SPM_ENCODING_ROOT / f"{split}.de",
        )
        
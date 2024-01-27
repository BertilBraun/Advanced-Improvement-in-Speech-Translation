import os

from src.mt.config import process_mt_dataset_to_spm_encoding
from src.asr.config import create_asr_configs
from src.asr.mel_encoding import process_dataset_to_mel_spectrogram
from src.datasets.base.asr_dataset import ASRDataset
from src.datasets.concrete.covost import CoVoST, CoVoSTPunctuationReconstructionDataset, CoVoSTWithText
from src.logger_utils import get_logger
from src.paths import *

logger = get_logger("Dataset::Prepare Datasets")

def can_skip_dataset_prep() -> bool:
    # if asr config file exists and mt spm data exists, we can skip
    config_file_exists = (ASR_COVOST_ROOT / "config.yaml").is_file()
    mt_spm_data_exists = all(
        (MT_COVOST_SPM_ENCODING_ROOT / f"{split}.en").is_file()
        for split in CoVoST.SPLITS
    )
    punctuation_spm_data_exists = all(
        (PUNCTUATION_COVOST_SPM_ENCODING_ROOT / f"{split}.en").is_file()
        for split in CoVoST.SPLITS
    )
    return config_file_exists and mt_spm_data_exists and punctuation_spm_data_exists

if __name__ == "__main__":
    if can_skip_dataset_prep():
        logger.info("Skipping dataset preparation, config file and MT spm data already exists")
        exit(0)
    
    logger.info("Preparing CoVoST...")
    covost_datasets = [CoVoST(COVOST_ROOT, split, "en", "de") for split in CoVoST.SPLITS]
    
    # copy spm model ASR_SPM_MODEL to ASR_COVOST_ROOT
    os.system(f"cp {ASR_SPM_MODEL} {ASR_COVOST_ROOT}")

    logger.info("Creating ASR configs for CoVoST...")
    create_asr_configs(
        covost_datasets,
        CoVoST.SPLITS,
        ASR_COVOST_ROOT,
        ASR_COVOST_MEL_ENCODING_ROOT,
        ASR_COVOST_MEL_ENCODING_ZIP,
        encoding_function=process_dataset_to_mel_spectrogram,
        spm_filename=ASR_SPM_MODEL.name,
    )    
    
    logger.info("Preparing MT CoVoST...")
    mt_datasets = [CoVoSTWithText(COVOST_ROOT, split, "en", "de") for split in CoVoST.SPLITS]
    
    for dataset in mt_datasets:
        process_mt_dataset_to_spm_encoding(dataset, MT_COVOST_ROOT, MT_SPM_MODEL)
    
    logger.info("Preparing MT CoVoST with punctuation...")
    punctuation_datasets = [CoVoSTPunctuationReconstructionDataset(COVOST_ROOT, split, "en", "de") for split in CoVoST.SPLITS]
    
    for dataset in punctuation_datasets:
        process_mt_dataset_to_spm_encoding(dataset, PUNCTUATION_COVOST_ROOT, MT_SPM_MODEL)
        
    logger.info("Done!")
    
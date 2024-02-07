import os

from src.logger_utils import get_logger

from src.datasets.concrete.covost import CoVoST, CoVoSTPunctuationReconstructionDataset, CoVoSTWithText
from src.preprocess.paraphrased_dataset import ParaphrasedDataset
from src.preprocess.punctuation_enhancement_dataset import PunctuationEnhancementDataset
from src.paths import *

logger = get_logger("Dataset::Prepare Datasets")

def can_skip_dataset_prep() -> bool:
    # if asr config file exists and mt spm data exists, we can skip
    mel_config_file_exists = (ASR_COVOST_ROOT / "config.yaml").is_file()
    wav2vec_config_file_exists = (ASR_COVOST_WAV2VEC_ROOT / "config.yaml").is_file() or True
    mt_spm_data_exists = all(
        (MT_COVOST_SPM_ENCODING_ROOT / f"{split}.en").is_file()
        for split in CoVoST.SPLITS
    )
    punctuation_spm_data_exists = all(
        (PUNCTUATION_COVOST_SPM_ENCODING_ROOT / f"{split}.en").is_file()
        for split in CoVoST.SPLITS
    )
    return mel_config_file_exists and wav2vec_config_file_exists and mt_spm_data_exists and punctuation_spm_data_exists

if __name__ == "__main__":
    if can_skip_dataset_prep():
        logger.info("Skipping dataset preparation, all config data already exists")
        exit(0)
        
    from src.mt.config import process_mt_dataset_to_spm_encoding
    from src.asr.config import create_asr_configs
    from src.asr.mel_encoding import process_dataset_to_mel_spectrogram
    # from src.asr.wav2vec_encoding import process_dataset_to_wav2vec_embeddings
    
    logger.info("Preparing CoVoST...")
    covost_datasets = [CoVoST(COVOST_ROOT, split, "en", "de") for split in CoVoST.SPLITS]
    
    # copy spm model ASR_SPM_MODEL to ASR_COVOST_ROOT
    os.system(f"cp {ASR_SPM_MODEL} {ASR_COVOST_ROOT}")

    logger.info("Creating ASR configs for CoVoST (mel)...")
    create_asr_configs(
        covost_datasets,
        CoVoST.SPLITS,
        ASR_COVOST_ROOT,
        ASR_COVOST_MEL_ENCODING_ROOT,
        encoding_function=process_dataset_to_mel_spectrogram,
        spm_filename=ASR_SPM_MODEL.name,
    )    
    
    # # copy spm model ASR_SPM_MODEL to ASR_COVOST_ROOT
    # os.system(f"cp {ASR_SPM_MODEL} {ASR_COVOST_WAV2VEC_ROOT}")
    # 
    # logger.info("Creating ASR configs for CoVoST (wav2vec)...")
    # create_asr_configs(
    #     covost_datasets,
    #     CoVoST.SPLITS,
    #     ASR_COVOST_WAV2VEC_ROOT,
    #     ASR_COVOST_WAV2VEC_ENCODING_ROOT,
    #     encoding_function=process_dataset_to_wav2vec_embeddings,
    #     spm_filename=ASR_SPM_MODEL.name,
    # )  
    
    logger.info("Preparing MT CoVoST...")
    mt_datasets = [CoVoSTWithText(COVOST_ROOT, split, "en", "de") for split in CoVoST.SPLITS]
    
    process_mt_dataset_to_spm_encoding(mt_datasets, CoVoST.SPLITS, MT_COVOST_ROOT, MT_SPM_MODEL)
    
    mt_datasets = [
        ParaphrasedDataset(mt_datasets[0]),
        mt_datasets[1],
        mt_datasets[2],        
    ]
    
    process_mt_dataset_to_spm_encoding(mt_datasets, CoVoST.SPLITS, MT_PARAPHRASED_COVOST_ROOT, MT_SPM_MODEL)
    
    logger.info("Preparing MT CoVoST with punctuation...")
    punctuation_datasets = [
        PunctuationEnhancementDataset(CoVoSTPunctuationReconstructionDataset(COVOST_ROOT, CoVoST.TRAIN, "en", "de")),
        CoVoSTPunctuationReconstructionDataset(COVOST_ROOT, CoVoST.VALID, "en", "de"),
        CoVoSTPunctuationReconstructionDataset(COVOST_ROOT, CoVoST.TEST, "en", "de"),
    ]
    
    process_mt_dataset_to_spm_encoding(punctuation_datasets, CoVoST.SPLITS, PUNCTUATION_COVOST_ROOT, PUNCTUATION_SPM_MODEL)
        
    logger.info("Done!")
    
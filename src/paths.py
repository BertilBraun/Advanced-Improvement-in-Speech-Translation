from pathlib import Path

COVOST = Path("/pfs/work7/workspace/scratch/ubppd-ASR/covost")
COVOST_ROOT = COVOST
COVOST_CORPUS = COVOST / "corpus-16.1"
COVOST_DATA = COVOST / "data"
COVOST_CORPUS_EN_CLIPS = COVOST_CORPUS / "en" / "clips"

MT_ROOT = Path("/pfs/work7/workspace/scratch/uxude-MT")
ASR_ROOT = Path("/pfs/work7/workspace/scratch/uxude-ASR")


MT_SPM_MODEL = MT_ROOT / "spm.mt.model"
ASR_SPM_MODEL = ASR_ROOT / "spm.asr.model"

ASR_COVOST_ROOT = ASR_ROOT / "dataset" / "covost"
ASR_COVOST_MEL_ENCODING_ROOT = ASR_COVOST_ROOT / "mel" / "encoded"
ASR_COVOST_MEL_ENCODING_ZIP = ASR_COVOST_ROOT / "mel" / "encoded.zip"

MT_COVOST_ROOT = MT_ROOT / "dataset" / "covost"
MT_COVOST_DATA_ROOT = MT_COVOST_ROOT / "data"
MT_COVOST_SPM_ENCODING_ROOT = MT_COVOST_ROOT / "spm"



for p in [
    COVOST_ROOT,
    COVOST_CORPUS,
    COVOST_DATA,
    COVOST_CORPUS_EN_CLIPS,
    MT_ROOT,
    ASR_ROOT,
    ASR_COVOST_ROOT,
    ASR_COVOST_MEL_ENCODING_ROOT,
    ASR_COVOST_MEL_ENCODING_ZIP,
    MT_COVOST_ROOT,
    MT_COVOST_DATA_ROOT,
    MT_COVOST_SPM_ENCODING_ROOT,
]:
    p.mkdir(parents=True, exist_ok=True)
    
assert MT_SPM_MODEL.is_file(), f"Please copy the MT spm model to this location: {MT_SPM_MODEL}"
assert ASR_SPM_MODEL.is_file(), f"Please copy the ASR spm model to this location: {ASR_SPM_MODEL}"

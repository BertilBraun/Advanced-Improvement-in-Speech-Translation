from pathlib import Path

COVOST_ROOT = Path('/pfs/work7/workspace/scratch/uxude-ASR/dataset/covost/en')
COVOST_CORPUS_EN_CLIPS = COVOST_ROOT / 'clips'

MT_ROOT = Path('/pfs/work7/workspace/scratch/uxude-MT')
ASR_ROOT = Path('/pfs/work7/workspace/scratch/uxude-ASR')


MT_SPM_MODEL = MT_ROOT / 'spm.mt.model'
ASR_SPM_MODEL = ASR_ROOT / 'spm.asr.model'
PUNCTUATION_SPM_MODEL = MT_ROOT / 'spm.punctuation.model'

ASR_COVOST_ROOT = ASR_ROOT / 'dataset' / 'covost'
ASR_COVOST_MEL_ENCODING_ROOT = ASR_COVOST_ROOT / 'mel' / 'encoded'

ASR_COVOST_WAV2VEC_ROOT = ASR_COVOST_ROOT / 'wav2vec'
ASR_COVOST_WAV2VEC_ENCODING_ROOT = ASR_COVOST_ROOT / 'wav2vec' / 'encoded'

MT_COVOST_ROOT = MT_ROOT / 'dataset' / 'covost'
MT_COVOST_DATA_ROOT = MT_COVOST_ROOT / 'data'
MT_COVOST_SPM_ENCODING_ROOT = MT_COVOST_ROOT / 'spm'

MT_PARAPHRASED_COVOST_ROOT = MT_ROOT / 'dataset' / 'covost_paraphrased'
MT_PARAPHRASED_COVOST_DATA_ROOT = MT_COVOST_ROOT / 'data'
MT_PARAPHRASED_COVOST_SPM_ENCODING_ROOT = MT_COVOST_ROOT / 'spm'


PUNCTUATION_COVOST_ROOT = MT_ROOT / 'dataset' / 'covost_punctuation_enhancement'
PUNCTUATION_COVOST_DATA_ROOT = PUNCTUATION_COVOST_ROOT / 'data'
PUNCTUATION_COVOST_SPM_ENCODING_ROOT = PUNCTUATION_COVOST_ROOT / 'spm'

COVOST_MT_PARAPHRASED_EN_FILE = MT_COVOST_DATA_ROOT / 'train_paraphrased.en'
COVOST_MT_PARAPHRASED_DE_FILE = MT_COVOST_DATA_ROOT / 'train_paraphrased.de'


ASR_TRAIN_WORKSPACE = ASR_ROOT / 'train' / 'finetune_asr_covost'
MT_TRAIN_WORKSPACE = MT_ROOT / 'train' / 'finetune_mt_covost_paraphrased'
PUNCTUATION_TRAIN_WORKSPACE = MT_ROOT / 'train' / 'train_punctuation_covost_enhancement'


for p in [
    COVOST_ROOT,
    COVOST_CORPUS_EN_CLIPS,
    MT_ROOT,
    ASR_ROOT,
    ASR_COVOST_ROOT,
    ASR_COVOST_MEL_ENCODING_ROOT,
    ASR_COVOST_WAV2VEC_ROOT,
    ASR_COVOST_WAV2VEC_ENCODING_ROOT,
    MT_COVOST_ROOT,
    MT_COVOST_DATA_ROOT,
    MT_COVOST_SPM_ENCODING_ROOT,
    PUNCTUATION_COVOST_ROOT,
    PUNCTUATION_COVOST_DATA_ROOT,
    PUNCTUATION_COVOST_SPM_ENCODING_ROOT,
]:
    p.mkdir(parents=True, exist_ok=True)

# assert MT_SPM_MODEL.is_file(), f"Please copy the MT spm model to this location: {MT_SPM_MODEL}"
# assert ASR_SPM_MODEL.is_file(), f"Please copy the ASR spm model to this location: {ASR_SPM_MODEL}"

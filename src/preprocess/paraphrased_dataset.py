import os
from src.datasets.concrete.covost import CoVoST
from src.logger_utils import get_logger

from src.datasets.base.mt_dataset import MTDataset, TextSample
from src.paths import COVOST_MT_PARAPHRASED_DE_FILE, COVOST_MT_PARAPHRASED_EN_FILE

logger = get_logger("Paraphrases::Paraphrased Dataset")

class ParaphrasedDataset(MTDataset):
    def __init__(self, base_dataset: MTDataset) -> None:
        self.base_dataset = base_dataset
        super().__init__(CoVoST.TRAIN)
        
    def _load_data(self) -> list[TextSample]:
        if not os.path.exists(COVOST_MT_PARAPHRASED_EN_FILE) or not os.path.exists(COVOST_MT_PARAPHRASED_DE_FILE):
            logger.error(f"Paraphrased files not found at {COVOST_MT_PARAPHRASED_EN_FILE} and {COVOST_MT_PARAPHRASED_DE_FILE}. Please run the paraphrase generation script.")
            return self.base_dataset._data
        
        with open(COVOST_MT_PARAPHRASED_EN_FILE, "r", encoding="utf-8") as en_file, \
             open(COVOST_MT_PARAPHRASED_DE_FILE, "r", encoding="utf-8") as de_file:
            return self.base_dataset._data + [(en, de) for en, de in zip(en_file, de_file)]

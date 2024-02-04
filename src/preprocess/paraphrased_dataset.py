from src.logger_utils import get_logger

from src.datasets.base.mt_dataset import MTDataset, TextSample
from src.preprocess.generate_paraphrases import OUTPUT_DE_FILE, OUTPUT_EN_FILE

logger = get_logger("Paraphrases::Paraphrased Dataset")

class ParaphrasedDataset(MTDataset):
    def __init__(self) -> None:
        super().__init__("train")
        
    def _load_data(self) -> list[TextSample]:
        with open(OUTPUT_EN_FILE, "r", encoding="utf-8") as en_file, \
             open(OUTPUT_DE_FILE, "r", encoding="utf-8") as de_file:
            return [(en, de) for en, de in zip(en_file, de_file)]

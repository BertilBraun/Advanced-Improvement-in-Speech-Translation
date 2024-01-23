from pathlib import Path

from examples.speech_to_text.data_utils import extract_fbank_features  # noqa

from src.datasets.asr_dataset import ASRDataset
from src.datasets.util import iterate_over_dataset
from src.logger_utils import get_logger

logger = get_logger("ASR::MelEncoding")


def process_dataset_to_mel_spectrogram(dataset: ASRDataset, output_root: Path) -> None:
    """Extracts mel spectrograms for the given dataset and saves them to the given output root as .npy files."""
    for wav, sample_rate, sentence, translation, speaker_id, sample_id in iterate_over_dataset(dataset, desc="Mel"):
        file = output_root / f"{speaker_id}-{sample_id}.npy"
        if not file.is_file():
            extract_fbank_features(wav, sample_rate, file)

    logger.info(f"Finished processing mel spectrogram for all samples.")

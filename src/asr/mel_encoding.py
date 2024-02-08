from pathlib import Path

from examples.speech_to_text.data_utils import extract_fbank_features

from src.datasets.base.asr_dataset import ASRDataset
from src.datasets.util import process_dataset_in_parallel
from src.logger_utils import get_logger

logger = get_logger('ASR::MelEncoding')


def process_dataset_to_mel_spectrogram(dataset: ASRDataset, output_root: Path) -> None:
    """Extracts mel spectrograms for the given dataset and saves them to the given output root as .npy files."""

    def process_sample(sample):
        wav, sample_rate, sentence, translation, speaker_id, sample_id = sample
        file = output_root / f'{speaker_id}-{sample_id}.npy'
        if not file.is_file():
            extract_fbank_features(wav, sample_rate, file)

    process_dataset_in_parallel(dataset, process_sample, desc='Mel')

    logger.info(f'Finished processing mel spectrogram for all samples.')

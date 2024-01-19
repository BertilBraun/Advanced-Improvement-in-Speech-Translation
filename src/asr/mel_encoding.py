from pathlib import Path

from examples.speech_to_text.data_utils import extract_fbank_features

from src.datasets.asr_dataset import ASRDataset
from src.datasets.util import iterate_over_dataset


def process_dataset_to_mel_spectrogram(dataset: ASRDataset, output_root: Path) -> None:
    """Extracts mel spectrograms for the given dataset and saves them to the given output root as .npy files."""
    for wav, sample_rate, _, spk_id, chapter_no, utt_no in iterate_over_dataset(
        dataset, desc="Mel"
    ):
        file = output_root / f"{spk_id}-{chapter_no}-{utt_no}.npy"
        if not file.is_file():
            extract_fbank_features(wav, sample_rate, file)

    print(f"Finished processing mel spectrogram for {len(dataset)} samples.")

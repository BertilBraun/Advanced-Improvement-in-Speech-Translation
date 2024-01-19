from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from src.datasets.asr_dataset import ASRDataset
from src.datasets.util import iterate_over_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

__PROCESSOR = None
__MODEL = None


def process_dataset_to_wav2vec_embeddings(
    dataset: ASRDataset, output_root: Path, batch_size: int = 32
) -> None:
    """Extracts wav2vec embeddings for the given dataset and saves them to the given output root as .npy files."""
    batch_waveforms = []
    batch_paths = []

    for wav, sample_rate, _, spk_id, chapter_no, utt_no in iterate_over_dataset(
        dataset, desc=f"Wav2vec"
    ):
        file = output_root / f"{spk_id}-{chapter_no}-{utt_no}.npy"

        if not file.is_file():
            batch_waveforms.append(wav.to(device=DEVICE, dtype=torch.float))
            batch_paths.append(file.as_posix())

        if len(batch_waveforms) == batch_size:
            __extract_wav2vec_features_batch(batch_waveforms, sample_rate, batch_paths)
            batch_waveforms = []
            batch_paths = []

    if batch_waveforms:
        __extract_wav2vec_features_batch(batch_waveforms, sample_rate, batch_paths)

    print(f"Finished  processing wav2vec embeddings for {len(dataset)} samples.")


def __ensure_model_and_processor_loaded() -> None:
    global __PROCESSOR
    global __MODEL

    if __PROCESSOR is None:
        print("Loading Wav2Vec processor...")
        __PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    if __MODEL is None:
        print("Loading Wav2Vec model...")
        __MODEL = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        __MODEL = __MODEL.to(DEVICE)


def __extract_wav2vec_features_batch(
    waveforms: list[torch.FloatTensor],
    sample_rate: int,
    output_paths: list[Path],
) -> None:
    __ensure_model_and_processor_loaded()

    try:
        # Store original lengths and remove the first dimension
        original_lengths = [wav.shape[1] for wav in waveforms]
        waveforms_squeezed = [wav.squeeze(0) for wav in waveforms]
        batch_waveforms = pad_sequence(waveforms_squeezed, batch_first=True)
        max_padded_length = batch_waveforms.shape[1]

        # Add a channel dimension
        batch_waveforms = batch_waveforms.unsqueeze(1)

        # Process the batch
        processed_values = __PROCESSOR(
            batch_waveforms, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values
        processed_values = processed_values.squeeze(0).squeeze(1)
        processed_values = processed_values.to(DEVICE)

        with torch.no_grad():
            features = __MODEL(processed_values).last_hidden_state
        feature_length = features.shape[1]

        # Trim the features based on original lengths and save
        for feature, original_length, output_path in zip(
            features, original_lengths, output_paths
        ):
            # Calculate the length ratio and apply it to the feature length
            length_ratio = original_length / max_padded_length
            trimmed_length = int(length_ratio * feature_length)

            # Optionally add a small buffer to avoid cutting off too much
            buffer = 5  # You can adjust this buffer
            trimmed_length = min(trimmed_length + buffer, feature_length)

            # Trim the feature
            trimmed_feature = feature.cpu().numpy()[:trimmed_length, :]
            np.save(output_path, trimmed_feature)
    except Exception as e:
        print(f"Error at embedding {output_paths}: {e}")
        return

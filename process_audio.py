from mpi4py import MPI
import torch
from torchaudio.datasets import LIBRISPEECH
import tqdm
from pathlib import Path

import torch
from pathlib import Path
import numpy as np
from typing import List
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# TODO adjust location for Cluster
# Something like: /pfs/work7/workspace/scratch/uxxxx-PST/speech_to_text/dataset/
DATASET_LOCATION = "/content/fairseq/examples/speech_to_text/data" 

# TODO adjust location for Cluster
# Something like: /pfs/work7/workspace/scratch/uxxxx-PST/speech_to_text/encoded_dataset/
ENCODED_OUTPUT_ROOT = DATASET_LOCATION + "/encoded"

# TODO Define your batch size
BATCH_SIZE = 12

wav2VecDevice = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Wav2Vec model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model = model.to(wav2VecDevice)

def extract_wav2vec_features_batch(
    waveforms: List[torch.FloatTensor],
    sample_rate: int,
    output_paths: List[Path],
    overwrite: bool = False
) -> None:
    # Check if all output files exist and if overwrite is not allowed
    if all(path.is_file() for path in output_paths) and not overwrite:
        return

    # Store original lengths and remove the first dimension
    original_lengths = [wav.shape[1] for wav in waveforms]
    waveforms_squeezed = [wav.squeeze(0) for wav in waveforms]
    batch_waveforms = pad_sequence(waveforms_squeezed, batch_first=True)
    max_padded_length = batch_waveforms.shape[1]

    # Add a channel dimension
    batch_waveforms = batch_waveforms.unsqueeze(1)

    # Process the batch
    processed_values = processor(batch_waveforms, sampling_rate=sample_rate, return_tensors="pt").input_values
    processed_values = processed_values.squeeze(0).squeeze(1)
    processed_values = processed_values.to(wav2VecDevice)

    with torch.no_grad():
        features = model(processed_values).last_hidden_state
    feature_length = features.shape[1]

    # Trim the features based on original lengths and save
    for feature, original_length, output_path in zip(features, original_lengths, output_paths):
        # Calculate the length ratio and apply it to the feature length
        length_ratio = original_length / max_padded_length
        trimmed_length = int(length_ratio * feature_length)

        # Optionally add a small buffer to avoid cutting off too much
        buffer = 5  # You can adjust this buffer
        trimmed_length = min(trimmed_length + buffer, feature_length)

        # Trim the feature
        trimmed_feature = feature.cpu().numpy()[:trimmed_length, :]
        np.save(output_path, trimmed_feature)

def process_dataset(dataset, feature_root) -> None:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    total_data = len(dataset)
    data_per_node = total_data // size

    start = rank * data_per_node
    end = (rank + 1) * data_per_node if rank != size - 1 else total_data

    batch_waveforms = []
    batch_paths = []

    for i in tqdm.tqdm(range(start, end), desc=f"Node {rank}"):
        wav, sample_rate, _, spk_id, chapter_no, utt_no = dataset[i]
        batch_waveforms.append(torch.FloatTensor(wav))
        batch_paths.append(feature_root / f"{spk_id}-{chapter_no}-{utt_no}.npy")

        if len(batch_waveforms) == BATCH_SIZE:
            extract_wav2vec_features_batch(batch_waveforms, sample_rate, batch_paths)
            batch_waveforms = []
            batch_paths = []

    if batch_waveforms:
        extract_wav2vec_features_batch(batch_waveforms, sample_rate, batch_paths)

    print(f"Node {rank} finished")

# Main execution
if __name__ == "__main__":
    # Define folder path to store the data
    out_root = Path(DATASET_LOCATION)
    # Create folder if not yet exist
    out_root.mkdir(exist_ok=True)
    
    print("Fetching training data...")
    train_data = LIBRISPEECH(out_root.as_posix(), url="train-clean-100", download=True) # TODO For final training set to 'train-clean-360'
    print("Fetching dev data...")
    dev_data = LIBRISPEECH(out_root.as_posix(), url="dev-clean", download=True)
    print("Fetching test data...")
    test_data = LIBRISPEECH(out_root.as_posix(), url="test-clean", download=True)

    feature_root =  Path(ENCODED_OUTPUT_ROOT)
    # Create folder if not yet exist
    feature_root.mkdir(exist_ok=True)
    
    for dataset in [train_data, dev_data, test_data]:
        process_dataset(dataset, feature_root)

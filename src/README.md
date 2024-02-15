# Source for shared code between scripts

## How is training structured?

### ASR structure

#### ASR training

1. Download the datasets. The dataset must implement the `Dataset` interface with a `__getitem__` method which returns a tuple of the form `(path, sample_rate, sentence, translation, speaker_id, sample_id)`.
2. Process the datasets to wav2vec embeddings or mel spectrograms in a `encodings` folder using the `mel_encoding.py` or `wav2vec_encoding.py` script.
3. Create the ASR configs using the `create_asr_configs` function in a `configs` folder.
4. Call `fairseq-train` with the created `configs` folder, set the `--save-dir` to a `checkpoints` folder and set `--train-subset` and `--valid-subset` to the appropriate dataset split names. These names are the same as the ones provided to the `create_asr_configs` function and are in the `configs` folder with a `.tsv` extension (e.g. `train.tsv`) as well.

#### ASR inference

- Call `fairseq-generate` with the created `configs` folder, set the `--gen-subset` to the appropriate dataset split name and set `--path` to the best checkpoint file in the `checkpoints` folder (e.g. `checkpoints/checkpoint_best.pt`).
  - The output format is on multiple lines, with the first character indicating the type of output on this line:
    - `H`: hypothesis
    - `T`: target (reference truth)
  - The last line of the output is the achieved score (whatever metric was used via `--scoring`).

#### Full ASR Example

```python
# download the datasets
COVOST_ROOT = "/pfs/work7/workspace/scratch/ubppd-ASR/covost"

covost = {
    split: CoVoST(COVOST_ROOT, split, "en", "de")
    for split in CoVoST.SPLITS
}

# process the datasets to wav2vec embeddings
for split, dataset in covost.items():
    process_dataset_to_wav2vec_embeddings(ASRDataset(dataset), Path(f"encodings/covost/{split}"))

# create the ASR configs
create_asr_configs(
    datasets=list(covost.values()),
    dataset_names=list(covost.keys()),
    root_location=Path("configs/covost"),
    encodings_folder=Path("encodings/covost"),
    zip_file=Path("configs/covost.zip"),
    vocab_size=5000,
    overwrite_zip=True,
)
```

```bash
# train the model
fairseq-train configs/covost --save-dir checkpoints/covost --train-subset train --valid-subset dev

# generate the output
fairseq-generate configs/covost --gen-subset test --path checkpoints/covost/checkpoint_best.pt
```

### MT structure

#### MT training

1. Download the datasets. The dataset must implement the `Dataset` interface with a `__getitem__` method which returns a tuple of the form `(sentence, translation)`.
2. Write out the datasets to a `data` folder by instantiating a `MTDataset` with the dataset and calling `MTDataset::write_to_files` with the appropriate dataset split name (e.g. `train`).
3. Train a byte pair encoding model using the `bpe.py` script.
4. Encode the datasets using the trained byte pair encoding model using the `bpe.py` script and save them to a `spm` folder.
5. Binarize the datasets using the `fairseq-preprocess` command with a `--destdir` of `binarized`. Here you also need to set the `--source-lang` and `--target-lang` to the appropriate language ids (e.g. `en` and `de`). These are automatically used to access the correct files in the `spm` folder (e.g. `spm/train.en` and `spm/train.de` when `--trainpref` is set to `spm/train`).
6. Call `fairseq-train` with the created `binarized` folder, set the `--save-dir` to a `checkpoints` folder.

#### MT inference

- If required, rerun `fairseq-preprocess` with the `--testpref` set to the appropriate dataset split name (e.g. `spm/tst.de-en`) so that the `binarized` folder contains the correct test data. This is only required if the test data is not already in the `binarized` folder (i.e. if we want to use a different test set than the one used for training). This requires the `--srcdict` and `--tgtdict` to be set to the dictionaries created during the initial `fairseq-preprocess` step.
- Call `fairseq-generate` with the created `binarized` folder, set the `--gen-subset` to the appropriate dataset split name and set `--path` to the best checkpoint file in the `checkpoints` folder (e.g. `checkpoints/checkpoint_best.pt`).
  - The output format is on multiple lines, with the first character indicating the type of output on this line:
    - `S`: source
    - `H`: hypothesis
    - `T`: target (reference truth)
  - The last line of the output is the achieved score (whatever metric was used via `--scoring`).

#### Full MT Example

```python
# download the datasets
WMTS = [14, 15, 16, 17, 18, 19] # list of all available wmt versions on huggingface

wmt = {
    split: WMT(WMTS, split, "en", "de")
    for split in WMT.SPLITS
}

# write out the datasets
for split, dataset in wmt.items():
    MTDataset(dataset).write_to_files(Path(f"data/wmt/{split}.en"), Path(f"data/wmt/{split}.de"))

# train the bpe model
bpe = BPE(retrain_spm=True, train_files=["data/wmt/train.en", "data/wmt/train.de"], vocab_size=15000)

# encode the datasets
for split in wmt.keys():
    bpe.encode_file(f"data/wmt/{split}.en", f"spm/{split}.en")
    bpe.encode_file(f"data/wmt/{split}.de", f"spm/{split}.de")

```

```bash
# binarize the datasets
fairseq-preprocess --source-lang en --target-lang de --trainpref spm/train --validpref spm/dev --testpref spm/tst --destdir binarized --workers 8

# train the model
fairseq-train binarized --save-dir checkpoints/wmt

# generate the output
fairseq-generate binarized --gen-subset test --path checkpoints/wmt/checkpoint_best.pt
```

## Functions

### Datasets

- `covost.py`

Provides 3 Classes which can be used to load the covost dataset.

1. `CoVoST` - Loads the dataset as a list of tuples of the form `(path, sample_rate, sentence, translation, speaker_id, sample_id)`.
2. `CoVoSTWithText` - Loads the dataset as a list of tuples of the form `(sentence, translation)` where the sentence and translation are loaded from the dataset without any audio for use in translation tasks.

Example usage:

```python
COVOST_ROOT = "/pfs/work7/workspace/scratch/ubppd-ASR/covost"
SOURCE_LANG_ID = "en"
TARGET_LANG_ID = "de"

for split in CoVoST.SPLITS:
    dataset = CoVoST(COVOST_ROOT, split, SOURCE_LANG_ID, TARGET_LANG_ID)
    for path, sample_rate, sentence, translation, speaker_id, sample_id in iterate_over_dataset(dataset):
        pass

    dataset_with_text = CoVoSTWithText(COVOST_ROOT, split, SOURCE_LANG_ID, TARGET_LANG_ID)
    for sentence, translation in iterate_over_dataset(dataset_with_text):
        pass
```

- `librispeech.py`

TODO - Should be implemented in the same way as `covost.py`.

- `wmt.py`

Provides a class `WMT` which can be used to load the wmt dataset.

Example usage:

```python
WMTS = [14, 15, 16, 17, 18, 19] # list of all available wmt versions on huggingface
MAX_DATASET_SIZE = 10_000_000 # limit the dataset size to 10 million samples

for split in WMT.SPLITS:
    print(f"Fetching split {split}...")
    dataset = WMT(WMTS, split, "en", "de", MAX_DATASET_SIZE)

    for en, de in iterate_over_dataset(dataset):
        pass
```

- `common_voice.py`

TODO - Should be implemented in the same way as `covost.py`. Should probably be used as comparison in the end.

- `mt_dataset.py`

Provides a class `MTDataset` which can be used to cleanup a loaded dataset.

Example usage:

```python
from src.datasets.wmt import WMT

datasource = WMT([19], "train", "en", "de")
dataset = MTDataset(datasource) # cleans up the dataset

for en, de in iterate_over_dataset(dataset):
    pass
```

- `asr_dataset.py`

Provides a class `ASRDataset` which given a dataset will return the audio and the sentence. The dataset must implement the `Dataset` interface with a `__getitem__` method which returns a tuple of the form `(path, sample_rate, sentence, translation, speaker_id, sample_id)`.

Example usage:

```python
from src.datasets.covost import CoVoST

COVOST_ROOT = "/pfs/work7/workspace/scratch/ubppd-ASR/covost"

datasource = CoVoST(COVOST_ROOT, "train", "en", "de")
dataset_with_audio = ASRDataset(datasource)

for waveform, sample_rate, sentence, translation, speaker_id, sample_id in iterate_over_dataset(dataset_with_audio):
    pass
```

- `util.py`

Provides a function `iterate_over_dataset(dataset: Dataset, desc="", start=0, end=None):` which can be used to iterate over a dataset with a progress bar and ignored errors.

Provides a class `ConcatenatedDatasets` which can be used to concatenate multiple datasets into one.

Example usage:

```python
from src.datasets.covost import CoVoSTwithText
from src.datasets.wmt import WMT
from src.datasets.util import ConcatenatedDatasets

COVOST_ROOT = "/pfs/work7/workspace/scratch/ubppd-ASR/covost"

covost = CoVoSTwithText(COVOST_ROOT, "train", "en", "de")
wmt = WMT([19], "train", "en", "de")

concatenated = ConcatenatedDatasets([covost, wmt])

for sentence, translation in iterate_over_dataset(concatenated):
    pass
```

### ASR

- `config.py`

Creates the ASR configs for the given datasets and saves them to the given root location.
If the zip file does not exist or overwrite_zip is True, the zip file will be created as well.
This function assumes that the datasets are already processed to wav2vec embeddings or mel spectrograms in the given encodings folder.
Assumes that the first dataset is the training dataset.

```python
def create_asr_configs(
    datasets: list[Dataset],
    dataset_names: list[str],
    root_location: Path,
    encodings_folder: Path,
    zip_file: Path, 
    vocab_size: int = 5000,
    overwrite_zip: bool = False,
    ) -> None:
    pass
```

- `mel_encoding.py`

Extracts mel spectrograms for the given dataset and saves them to the given output root as .npy files.
The dataset must be an `ASRDataset` which returns the audio and the sentence.

```python
def process_dataset_to_mel_spectrogram(dataset: ASRDataset, output_root: Path) -> None:
    pass
```

- `wav2vec_encoding.py`

Extracts wav2vec embeddings for the given dataset and saves them to the given output root as .npy files.
The dataset must be an `ASRDataset` which returns the audio and the sentence.

```python
def process_dataset_to_wav2vec_embeddings(dataset: ASRDataset, output_root: Path, batch_size: int = 32) -> None:
    pass
```

### MT

- `bpe.py`

Provides a class `BPE` which can be used to train a byte pair encoding model and encode/decode files.

Example usage:

```python
bpe = BPE(retrain_spm=True, train_files=["data/train.en", "data/train.de"], vocab_size=15000)

bpe.encode_file("data/train.en", "data/spm.train.en")

bpe.decode_file("data/spm.train.en", "data/train.en")
```

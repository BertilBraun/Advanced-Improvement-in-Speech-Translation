
import os
import sentencepiece as spm

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)

HOME_FOLDER = Path(f"{os.environ['HOME']}")
ROOT_FOLDER = HOME_FOLDER / "MT"

DATASET_FOLDER = ROOT_FOLDER / "dataset"
DATASET_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FOLDER = ROOT_FOLDER / "output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

DATASET_EN = DATASET_FOLDER / "train.de-en.en"
DATASET_DE = DATASET_FOLDER / "train.de-en.de"

PARAPHRASED_EN_FILE = OUTPUT_FOLDER / "train_paraphrased.de-en.en"
PARAPHRASED_DE_FILE = OUTPUT_FOLDER / "train_paraphrased.de-en.de"

OUTPUT_EN_FILE = OUTPUT_FOLDER / "train_complete.de-en.en"
OUTPUT_DE_FILE = OUTPUT_FOLDER / "train_complete.de-en.de"

SPM_OUTPUT_EN_FILE = OUTPUT_FOLDER / "spm.train_complete.de-en.en"
SPM_OUTPUT_DE_FILE = OUTPUT_FOLDER / "spm.train_complete.de-en.de"

LOG_FILE = OUTPUT_FOLDER / "paraphrases.log"
SEPARATOR = "  SEPARATOR  "


def data_generator():
    # WMT 14-19 datasets, but generate a set from these to avoid duplicates
    s = set()    
    total_sizes = 0
    
    for i in range(14, 15):
        log(f"Loading WMT{i} dataset...")
        dataset = load_dataset(f"wmt{i}", "de-en", split="train", trust_remote_code=True)
        log(f"Now processing WMT{i} dataset...")
        log(f"Dataset length: {len(dataset['translation'])}")
        total_sizes += len(dataset["translation"])
        
        # extend the set with the new sentences
        s.update(
            (sentence["en"], sentence["de"]) for sentence in dataset["translation"]
        )

    # convert the set to a list
    s = list(s)
    log(f"The total size of the dataset is {total_sizes}.")
    log(f"Filtered dataset length: {len(s)}")
    log(f"Removed {total_sizes - len(s)} duplicates.")
    
    return s

s = set()

def translation_pair_check(en, de):
    global s
    # only keep sentences that are only ascii characters
    for sentence in (en, de):
        for i, c in enumerate(sentence):
            if ord(c) >= 256 and c not in "–“’‘„”":
                s.add((sentence, i, c, ord(c)))
                #print(f"{sentence} at index {i} with character {c} ord(c): {ord(c)}) is not ascii.")
                return False
    return True


if __name__ == "__main__":
    
    skipped_lines = 0
    log("Merging paraphrased and original datasets...")
    
    with open(OUTPUT_DE_FILE, "w", encoding="utf-8") as f_de, \
        open(OUTPUT_EN_FILE, "w", encoding="utf-8") as f_en, \
        open(PARAPHRASED_EN_FILE, "r", encoding="utf-8") as f_paraphrased_en, \
        open(PARAPHRASED_DE_FILE, "r", encoding="utf-8") as f_paraphrased_de:
        
        for en, de in tqdm(zip(f_paraphrased_en.readlines(), f_paraphrased_de.readlines()), desc="Merging datasets"):
            if translation_pair_check(en, de):
                f_en.write(en)
                f_de.write(de)
            else:
                skipped_lines += 1
                print(en, de)
            
        for en, de in tqdm(data_generator(), desc="Merging WMT datasets"):
            if translation_pair_check(en, de):
                f_en.write(en)
                f_de.write(de)
            else:
                skipped_lines += 1
                print(en, de)
                
    log(f"Skipped {skipped_lines} lines because they contained non-ascii characters.")
    log("Non-ascii characters:")
    for sentence, i, c, ord_c in s:
        log(c, ord_c)

    spm.SentencePieceTrainer.train(input=f"{OUTPUT_DE_FILE},{OUTPUT_EN_FILE}",
                                model_prefix="bpe",
                                vocab_size=5000)

    log('Finished training sentencepiece model.')
    
    spm_model = spm.SentencePieceProcessor(model_file="bpe.model")
    
    log("BPE model ready.")
    
    for data, spm in zip((OUTPUT_DE_FILE, OUTPUT_EN_FILE), (SPM_OUTPUT_DE_FILE, SPM_OUTPUT_EN_FILE)):
        
        with open(data, "r", encoding="utf-8") as f_in, \
            open(spm, "w", encoding="utf-8") as f_out:
            for line in tqdm(f_in.readlines(), desc=f"Segmenting '{data}' Dataset"):
                # Segmented into subwords
                line_segmented = spm_model.encode(line.strip(), out_type=str)
                f_out.write(" ".join(line_segmented) + "\n")
from pathlib import Path
import time
from typing import Literal, Union

import itertools
import os
import regex
import sentencepiece as spm
from tqdm import tqdm
from llama_cpp import Llama
from datasets import load_dataset
import spacy
import random

HOME_FOLDER = Path(f"{os.environ['HOME']}")
ROOT_FOLDER = HOME_FOLDER / "MT"

DATASET_FOLDER = ROOT_FOLDER / "dataset"
DATASET_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FOLDER = ROOT_FOLDER / "output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

DATASET_EN = DATASET_FOLDER / "train.de-en.en"
DATASET_DE = DATASET_FOLDER / "train.de-en.de"

OUTPUT_EN_FILE = OUTPUT_FOLDER / "train_paraphrased.de-en.en"
OUTPUT_DE_FILE = OUTPUT_FOLDER / "train_paraphrased.de-en.de"

DATASET_URL = "https://bwsyncandshare.kit.edu/s/7oo2AG8jRriLZKg/download?path=%2F&files=data.zip&downloadStartSecret=tk6qdncox5"

MODEL_PATH = HOME_FOLDER / "ST/llama-2-7b.Q5_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q5_K_M.gguf"

# Smaller apparently faster model 
MODEL_PATH = HOME_FOLDER / "ST/llama-2-7b.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf"

LANGUAGE = Union[Literal["en"], Literal["de"]]

PROMPT = {  # TODO Modify and play with this prompt to properly generate 5 good paraphrases
    "en": "Generate five distinct paraphrases of the following English sentence:\n'{}'\nParaphrases:",
    "de": "Schreiben Sie den folgenden Satz auf fünf verschiedene Arten um:\n'{}'\nUmschreibungen:",
}

# Some alternatives to try for paraphrase genaration
"""
PROMPT_Variation = {
    'en': "Provide five alternative expressions for the given sentence:\n'{}'\nAlternatives:",
    'de': "Geben Sie fünf alternative Ausdrücke für den folgenden Satz an:\n'{}'\nAlternativen:"
}

PROMPT_Rephrase = {
    'en': "Rephrase the following sentence in five different ways:\n'{}'\nRephrases:",
    'de': "Umschreiben Sie den folgenden Satz auf fünf verschiedene Arten:\n'{}'\nUmschreibungen:"
}

PROMPT_Diversify = {
    'en': "Diversify the expression of the given sentence in five ways:\n'{}'\nExpressions:",
    'de': "Diversifizieren Sie den Ausdruck des folgenden Satzes auf fünf Arten:\n'{}'\nAusdrücke:"
}

PROMPT_AlternateVersions = {
    'en': "Generate five alternate versions of the following sentence:\n'{}'\nVersions:",
    'de': "Erstellen Sie fünf alternative Versionen des folgenden Satzes:\n'{}'\nVersionen:"
}

PROMPT_Rewrite = {
    'en': "Rewrite the following sentence in five different ways:\n'{}'\nRewrites:",
    'de': "Schreiben Sie den folgenden Satz auf fünf verschiedene Arten um:\n'{}'\nUmschreibungen:"
}
"""

try:
    NLP = {"en": spacy.load("en_core_web_sm"), "de": spacy.load("de_core_news_sm")}
except OSError:
    print("Downloading Spacy models...")
    os.system("python -m spacy download en")
    os.system("python -m spacy download de")
    NLP = {"en": spacy.load("en_core_web_sm"), "de": spacy.load("de_core_news_sm")}
    

def ensure_model_loaded() -> None:
    # if model not found, download it
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Downloading...")
        download_command = f"wget -nc -O '{MODEL_PATH}' '{MODEL_URL}'"
        os.system(download_command) 
        print("Model downloaded.")
    print(f"Model ready at '{MODEL_PATH}'.")


def ensure_dataset_loaded() -> None:
    if not DATASET_EN.is_file() or not DATASET_DE.is_file():

        download_location = DATASET_FOLDER / "data.zip"
        # Download and unzip command
        download_command = f"wget -nc -O '{download_location}' '{DATASET_URL}'"
        unzip_command = f"unzip -o '{download_location}' -d '{DATASET_FOLDER}'"

        # Execute download and unzip commands
        print("Checking and downloading dataset...")
        os.system(download_command)
        print("Unzipping dataset...")
        os.system(unzip_command)
        print("Dataset ready.")


# Generates 5 paraphrases for the input sentence with LLaMA 2
def generate_paraphrases(LLM, sentence: str, language: LANGUAGE) -> list[str]:
    print(f"Generating paraphrases for '{sentence}' in {language}...")

    # Format the prompt with the given sentence
    formatted_prompt = PROMPT[language].format(sentence)

    # Generate response using LLaMA 2
    # max_tokens=0 removes the response size limit
    output = LLM(formatted_prompt, max_tokens=0)

    # Extract paraphrases from the response
    paraphrases_text = output["choices"][0]["text"]
    
    print(f"Paraphrases text: {paraphrases_text}")

    # Use Spacy's sentence boundary detection to extract paraphrases
    # doc = NLP[language](paraphrases_text)
    # sentences = [sent.text.strip() for sent in doc.sents]
    sentences = [
        cleanup_paraphrase(sent) for sent in paraphrases_text.split("\n") if cleanup_paraphrase(sent)
    ]

    # Consider you have to generate initally more paraphrases if you use diversify with diversity_factor<1
    sentences = diversify_paraphrases(
        sentences, diversity_factor=1
    )  # adjust diversity factor appropriately

    # Heuristics to identify and extract paraphrases
    paraphrases = [
        sent for sent in sentences if heuristic_is_paraphrase(sent, sentence, language)
    ]
    non_paraphrases = [
        sent for sent in sentences if sent not in paraphrases
    ]
    print(f"Non paraphrases: {non_paraphrases}")

    if len(paraphrases) < 5:
        print(
            f"Warning: Only {len(paraphrases)} paraphrases generated for '{sentence}'"
        )
        
    print(f"Possible paraphrases: {paraphrases}")

    # Ensure only five paraphrases are returned
    return paraphrases[:5]


def cleanup_paraphrase(sent: str) -> str:
    sent = sent.strip()
    
    # Remove leading and trailing quotation marks
    if sent.startswith('"') and sent.endswith('"'):
        sent = sent[1:-1]
    
    # Remove leading enumeration "\d+[\.:]"
    regex_enumeration = r"^\d+[\.:]"
    sent = regex.sub(regex_enumeration, "", sent)
    
    # Remove leading and trailing whitespace
    sent = sent.strip()
    
    # Remove leading and trailing quotation marks again, as they might have only been around the text without the enumeration
    if sent.startswith('"') and sent.endswith('"'):
        sent = sent[1:-1]
        
    sent = sent.strip()
    
    return sent    


# Diversify the list of paraphrases to cover a broader range of semantic variations
# If you would like to work with 5 paraphrases and use diversify_paraphrases with a
# diversity_factor<1 you have to generate initially more paraphrases
def diversify_paraphrases(paraphrases, diversity_factor=0.5):
    num_paraphrases = len(paraphrases)
    num_to_keep = max(int(num_paraphrases * diversity_factor), 1)

    # Randomly select paraphrases to keep
    diversified_paraphrases = random.sample(paraphrases, num_to_keep)

    return diversified_paraphrases


def heuristic_is_paraphrase(candidate: str, original: str, language: LANGUAGE) -> bool:
    # Convert sentences to Spacy Doc objects
    doc_candidate = NLP[language](candidate)
    doc_original = NLP[language](original)

    # Semantic similarity (using Spacy's built-in similarity method)
    similarity_threshold = 0.4  # TODO Adjust this threshold as needed
    if doc_candidate.similarity(doc_original) < similarity_threshold:
        return False

    # Structural or lexical difference
    if (
        candidate.lower() == original.lower()
        or doc_candidate.text_with_ws == doc_original.text_with_ws
    ):
        return False

    # Length similarity (within a certain percentage difference)
    length_diff_threshold = 0.4  # TODO 40% length difference threshold
    if (
        abs(len(candidate) - len(original)) / max(len(candidate), len(original))
        > length_diff_threshold
    ):
        return False

    return True


def read_dataset_segment(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as file:
        return [ line.strip() for line in file ]


def main() -> None:
    ensure_dataset_loaded()
    ensure_model_loaded()
     
    datasets = [
        load_dataset(f"wmt{i}", "de-en", split="train", trust_remote_code=True)
        for i in range(14, 20)
    ]
        
    # LLaMA model initialization
    LLM = Llama(model_path=MODEL_PATH.as_posix(), n_ctx=2048)

    # Read the dataset segment for this process
    english_sentences = read_dataset_segment(DATASET_EN) + [
            sentence["en"]
            for dataset in datasets
            for sentence in dataset["translation"]
        ]
    german_sentences = read_dataset_segment(DATASET_DE) + [
            sentence["de"]
            for dataset in datasets
            for sentence in dataset["translation"]
        ]
     
    print(f"Generating paraphrases for {len(english_sentences)} sentence pairs...")
    if len(english_sentences) != len(german_sentences):
        print("Warning: English and German sentences are not the same length.")

    with open(OUTPUT_EN_FILE, "w", encoding="utf-8") as en_file,\
        open(OUTPUT_DE_FILE, "w", encoding="utf-8") as de_file:
        # For each sentence pair in the dataset segment, generate paraphrases and write all combinations to files
        for en, de in zip(english_sentences, german_sentences):
            start = time.time()
            en_paraphrases = [en] + generate_paraphrases(LLM, en, "en")
            de_paraphrases = [de] + generate_paraphrases(LLM, de, "de")
            print(f"Paraphrases generated in {time.time() - start} seconds.")

            # Generate all combinations of English and German paraphrases and write directly to files
            for en_p, de_p in itertools.product(en_paraphrases, de_paraphrases):
                en_file.write(f"{en_p}\n")
                de_file.write(f"{de_p}\n")
           
    print("Finished generating paraphrases.")


    spm.SentencePieceTrainer.train(input=f"{OUTPUT_DE_FILE},{OUTPUT_EN_FILE}",
                                model_prefix="bpe",
                                vocab_size=5000)

    print('Finished training sentencepiece model.')
    
    spm_model = spm.SentencePieceProcessor(model_file="bpe.model")

    for lang in ["de", "en"]:
        with open(f"{DATASET_FOLDER}/train_paraphrased.de-en.{lang}", "r") as f_in, \
                open(f"{DATASET_FOLDER}/spm.train_paraphrased.de-en.{lang}", "w") as f_out:
            for line in tqdm(f_in.readlines(), desc=f"Segmenting {lang}"):
                # Segmented into subwords
                line_segmented = spm_model.encode(line.strip(), out_type=str)
                # Join the subwords into a string
                line_segmented = " ".join(line_segmented)
                f_out.write(line_segmented + "\n")


if __name__ == "__main__":
    main()

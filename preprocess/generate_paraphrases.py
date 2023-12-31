from pathlib import Path
import time
from typing import Literal, Union

import itertools
import os
import regex
import sentencepiece as spm
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import torch
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
    

print("Loading LLaMA...")

# TODO I think this is llama 1 and not llama 2
# TODO from here (https://www.mlexpert.io/machine-learning/tutorials/alpaca-and-llama-inference)
# TODO ref for llama 2 here (https://huggingface.co/blog/llama2)
LLAMA_MODEL = "baffo32/decapoda-research-llama-7B-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOKENIZER = LlamaTokenizer.from_pretrained(LLAMA_MODEL)
 
LLM = LlamaForCausalLM.from_pretrained(
    LLAMA_MODEL,
    # load_in_8bit=True,
    device_map="auto",
)
LLM.config.pad_token_id = TOKENIZER.pad_token_id = 0  # unk
LLM.config.bos_token_id = 1
LLM.config.eos_token_id = 2
 
LLM = LLM.eval()
LLM = torch.compile(LLM)
LLM = LLM.to(DEVICE)
print("LLaMA ready.")


def generate(prompt: str) -> str:
    encoding = TOKENIZER(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"].to(DEVICE)
 
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        repetition_penalty=1.1,
    )
    # generation_config = GenerationConfig( # Greedy
    #     num_beams=1,
    #     do_sample=False,
    # )

    with torch.inference_mode():
        encoded_output = LLM.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=64, # TODO increase this if you want longer paraphrases
        )
    
    decoded_output = TOKENIZER.decode(encoded_output.sequences[0])
    response = decoded_output.replace(prompt, "").strip()
    
    return response

# Generates 5 paraphrases for the input sentence with LLaMA 2
def generate_paraphrases(sentence: str, language: LANGUAGE) -> list[str]:
    print(f"Generating paraphrases for '{sentence}' in {language}...")

    # Format the prompt with the given sentence
    formatted_prompt = PROMPT[language].format(sentence)

    # Generate response using LLaMA 2
    # max_tokens=0 removes the response size limit
    # output = LLM(formatted_prompt, max_tokens=0)

    # Extract paraphrases from the response
    # paraphrases_text = output["choices"][0]["text"]
    paraphrases_text = generate(formatted_prompt)
    
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
    for quote in ['"', "'"]:
        if sent.startswith(quote) and sent.endswith(quote):
            sent = sent[1:-1]
    
    # Remove leading enumeration "\d+[\.:]"
    regex_enumeration = r"^\d+[\.:]"
    sent = regex.sub(regex_enumeration, "", sent)
    
    # Remove leading and trailing whitespace
    sent = sent.strip()
    
    # Remove leading and trailing quotation marks again, as they might have only been around the text without the enumeration
    for quote in ['"', "'"]:
        if sent.startswith(quote) and sent.endswith(quote):
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


def data_generator():
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

    ensure_dataset_loaded()
    
    # Read the dataset segment for this process
    english_sentences = read_dataset_segment(DATASET_EN)
    german_sentences = read_dataset_segment(DATASET_DE)
     
    if len(english_sentences) != len(german_sentences):
        print("Warning: English and German sentences are not the same length.")
    
    print(f"Original dataset length: {len(english_sentences)}")
    
    yield from zip(english_sentences, german_sentences)
    
    for i in range(14, 20):
        print(f"Loading WMT{i} dataset...")
        dataset = load_dataset(f"wmt{i}", "de-en", split="train", trust_remote_code=True)
        print(f"Now processing WMT{i} dataset...")
        print(f"Dataset length: {len(dataset['translation'])}")
        
        yield from zip(
            (sentence["en"] for sentence in dataset["translation"]),
            (sentence["de"] for sentence in dataset["translation"])
        )
     

def main() -> None:
    print("Generating paraphrases for all sentence pairs...")

    with open(OUTPUT_EN_FILE, "w", encoding="utf-8") as en_file,\
        open(OUTPUT_DE_FILE, "w", encoding="utf-8") as de_file:
        # For each sentence pair in the dataset segment, generate paraphrases and write all combinations to files
        for en, de in data_generator():
            start = time.time()
            print(f"Generating paraphrases for '{en}' and '{de}'...", flush=True)
            en_paraphrases = [en] + generate_paraphrases(en, "en")
            de_paraphrases = [de] + generate_paraphrases(de, "de")
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

from pathlib import Path
import time
from typing import Literal, Union

import itertools
import os
import regex
import sentencepiece as spm
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
import torch
from datasets import load_dataset
import spacy

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

LOG_FILE = OUTPUT_FOLDER / "paraphrases.log"
SEPARATOR = "  SEPARATOR  "

DATASET_URL = "https://bwsyncandshare.kit.edu/s/7oo2AG8jRriLZKg/download?path=%2F&files=data.zip&downloadStartSecret=tk6qdncox5"

LANGUAGE = Union[Literal["en"], Literal["de"]]

PROMPT = {  # TODO Modify and play with this prompt to properly generate 5 good paraphrases
    "en": "Generate five distinct paraphrases of the following English sentence:\n'{}'\nParaphrases:",
    "de": "Schreiben Sie den folgenden Satz auf fünf verschiedene Arten auf Deutsch um:\n'{}'\nUmschreibungen:",
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

LLAMA_MODEL = {
    "en": "meta-llama/Llama-2-7b-chat-hf",
    "de": "jphme/em_german_mistral_v01", # "em_german_7b_v01",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print("Loading Tokenizer")

# TOKENIZER = {
#     "en": AutoTokenizer.from_pretrained(LLAMA_MODEL["en"]),
#     "de": AutoTokenizer.from_pretrained(LLAMA_MODEL["de"]),
# }
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL["en"], padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
TOKENIZER = {
    "en": tokenizer,
    "de": tokenizer,
}

print("Tokenizer ready.")
print("Loading LLaMA model.")
 
# LLM = {
#     "en": AutoModelForCausalLM.from_pretrained(LLAMA_MODEL["en"]), #, low_cpu_mem_usage=True, load_in_8bit=True),
#     "de": AutoModelForCausalLM.from_pretrained(LLAMA_MODEL["de"]), #, low_cpu_mem_usage=True, load_in_8bit=True),
# }

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

llama = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL["en"], quantization_config=bnb_config)
LLM = {
    "en": llama,
    "de": llama,
}

print("Configuring LLaMA model.")

for language, model in LLM.items():
    model.config.pad_token_id = TOKENIZER[language].pad_token_id = 0  # unk
    model.config.bos_token_id = 1  # bos
    model.config.eos_token_id = 2  # eos
    
    model = model.eval()
    model = torch.compile(model)
    model = model.to(DEVICE)

print("LLaMA ready.")

BASE_PROMPT_TOKEN_LENGTH = {
    "en": len(TOKENIZER["en"].encode(PROMPT["en"].format(""))) - 1,
    "de": len(TOKENIZER["de"].encode(PROMPT["de"].format(""))) - 1,
}

print(f"Base prompt token length: {BASE_PROMPT_TOKEN_LENGTH}")

# 5 is the number of paraphrases to generate 
# 1.5 states that the paraphrase can be 65% longer than the input sentence
PROMPT_LENGTH_MULTIPLIER = 1.65 * 5 

PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
You are a professional writer.
<</SYS>>

{} [/INST]"""


# TODO can paraphrase generation be batched? https://github.com/huggingface/transformers/issues/25353
def generate(prompts: list[str], lng: LANGUAGE) -> list[str]:
    model_inputs = TOKENIZER[lng](prompts, return_tensors="pt", padding=True).to(DEVICE)
    
    max_input_length = model_inputs.input_ids.shape[-1]
    max_new_tokens = BASE_PROMPT_TOKEN_LENGTH[lng] + (max_input_length - BASE_PROMPT_TOKEN_LENGTH[lng]) * PROMPT_LENGTH_MULTIPLIER
 
    generation_config = GenerationConfig(
        temperature=0.2, # TODO really low temperature but still get hallucinations
        top_p=0.75,
        repetition_penalty=1.1,
    )
    # generation_config = GenerationConfig( # Greedy
    #     num_beams=1,
    #     do_sample=False,
    # )

    with torch.inference_mode():
        generated_ids = LLM[lng].generate(
            **model_inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    
    decoded_outputs = TOKENIZER[lng].batch_decode(generated_ids, skip_special_tokens=True)
    
    return [
        output.replace(prompt, "").strip()
        for output, prompt in zip(decoded_outputs, prompts)
    ]

# Generates 5 paraphrases for the input sentence with LLaMA 2
def generate_paraphrases(sentence: str, language: LANGUAGE) -> list[str]:
    if len(sentence) > 100: # TODO having to skip long sentences because of hallucinations
        print(f"Warning: Sentence '{sentence}' is is too long for paraphrase generation. Skipping.")
        return [sentence]
    
    print(f"\nGenerating paraphrases for '{sentence}' in {language}...")

    # Format the prompt with the given sentence
    formatted_prompt = PROMPT_TEMPLATE.format(PROMPT[language].format(sentence))

    # Generate response using LLaMA 2
    paraphrases_text = generate([formatted_prompt], language)[0]
    
    print(f"Paraphrases text: \"\"\"{paraphrases_text}\"\"\"")

    sentences = list(set((
        cleanup_paraphrase(sent) for sent in paraphrases_text.split("\n") if cleanup_paraphrase(sent)
    )))

    # Heuristics to identify and extract paraphrases
    paraphrases = [
        sent for sent in sentences if heuristic_is_paraphrase(sent, sentence, language)
    ]
    non_paraphrases = [
        sent for sent in sentences if sent not in paraphrases
    ]
    print("Non paraphrases:")
    for sent in non_paraphrases:
        print(f"    {sent}")

    if len(paraphrases) < 5:
        print(f"Warning: Only {len(paraphrases)} paraphrases generated for '{sentence}'")
        
    print("Paraphrases:")
    for sent in paraphrases:
        print(f"    {sent}")

    # Ensure only five paraphrases are returned
    return paraphrases # TODO currently return all [:5]


def cleanup_paraphrase(sent: str) -> str:
    sent = sent.replace("<s>", "").replace("</s>", "").strip()
    
    # Remove leading and trailing quotation marks
    for quote in ['"', "'"]:
        if sent.startswith(quote) and sent.endswith(quote):
            sent = sent[1:-1]
            
    sent = sent.strip()
    
    # Remove leading enumeration "1." or "1:" or "1)" or "(1)"
    regex_enumeration = r"^\s*(\d+[\.:)]|\(\d+\))"
    sent = regex.sub(regex_enumeration, "", sent)
    
    # Remove leading enumeration "-*."    
    for start in "-*.":
        if sent.startswith(start):
            sent = sent[1:]
            
    # Remove leading emumeration "a)" or "a." or "a:" or (a)" but only for one or two letters
    regex_enumeration = r"^(\s*[a-zA-Z]{1,2}[\.:)]|\([a-zA-Z]{1,2}\))"
    sent = regex.sub(regex_enumeration, "", sent)
    
    # Remove leading and trailing whitespace
    sent = sent.strip()
    
    # Remove leading and trailing quotation marks again, as they might have only been around the text without the enumeration
    for quote in ['"', "'"]:
        if sent.startswith(quote) and sent.endswith(quote):
            sent = sent[1:-1]
        
    sent = sent.strip()
    
    return sent    


def heuristic_is_paraphrase(candidate: str, original: str, language: LANGUAGE) -> bool:
    # Convert sentences to Spacy Doc objects
    doc_candidate = NLP[language](candidate)
    doc_original = NLP[language](original)

    # Semantic similarity (using Spacy's built-in similarity method)
    similarity_threshold = 0.2  # TODO Adjust this threshold as needed
    if doc_candidate.similarity(doc_original) < similarity_threshold:
        return False

    # Structural or lexical difference
    if (
        candidate.lower() == original.lower()
        or doc_candidate.text_with_ws == doc_original.text_with_ws
    ):
        return False

    # Length similarity (within a certain percentage difference)
    length_diff_threshold = 0.65  # TODO 65% length difference threshold
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
        open(OUTPUT_DE_FILE, "w", encoding="utf-8") as de_file, \
        open(LOG_FILE, "a", encoding="utf-8") as log_file:
        # For each sentence pair in the dataset segment, generate paraphrases and write all combinations to files
        for en, de in data_generator():
            start = time.time()
            print(f"\n\nGenerating paraphrases for '{en}' and '{de}'...", flush=True)
            en_paraphrases = [en] + generate_paraphrases(en, "en")
            de_paraphrases = [de] + generate_paraphrases(de, "de")
            print(f"Paraphrases generated in {round(time.time() - start, 2)} seconds.")

            # Generate all combinations of English and German paraphrases and write directly to files
            for en_p, de_p in itertools.product(en_paraphrases, de_paraphrases):
                en_file.write(f"{en_p}\n")
                de_file.write(f"{de_p}\n")
                
            log_file.write(SEPARATOR.join([en, de, *en_paraphrases, *de_paraphrases]) + "\n")
           
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

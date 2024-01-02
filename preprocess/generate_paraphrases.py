import json
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

OUTPUT_EN_FILE = OUTPUT_FOLDER / "train_paraphrased.de-en.en"
OUTPUT_DE_FILE = OUTPUT_FOLDER / "train_paraphrased.de-en.de"

SPM_OUTPUT_FILE = OUTPUT_FOLDER / "spm.train_paraphrased.de-en.{}"

LOG_FILE = OUTPUT_FOLDER / "paraphrases.log"
SEPARATOR = "  SEPARATOR  "

DATASET_URL = "https://bwsyncandshare.kit.edu/s/7oo2AG8jRriLZKg/download?path=%2F&files=data.zip&downloadStartSecret=tk6qdncox5"

LANGUAGE = Union[Literal["en"], Literal["de"]]

try:
    NLP = {"en": spacy.load("en_core_web_sm"), "de": spacy.load("de_core_news_sm")}
except OSError:
    log("Downloading Spacy models...")
    os.system("python -m spacy download en")
    os.system("python -m spacy download de")
    NLP = {"en": spacy.load("en_core_web_sm"), "de": spacy.load("de_core_news_sm")}
    

log("Loading LLaMA...")

LLAMA_MODEL = {
    "en": "meta-llama/Llama-2-7b-chat-hf",
    "de": "jphme/em_german_mistral_v01", # "em_german_7b_v01",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

log(f"Using device: {DEVICE}")
log("Loading Tokenizer")

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

log("Tokenizer ready.")
log("Loading LLaMA model.")
 
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

llama = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL["en"], quantization_config=bnb_config, device_map="auto")
LLM = {
    "en": llama,
    "de": llama,
}

log("Configuring LLaMA model.")

for language, model in LLM.items():
    model.config.pad_token_id = TOKENIZER[language].pad_token_id = 0  # unk
    model.config.bos_token_id = 1  # bos
    model.config.eos_token_id = 2  # eos
    
    # TODO test with this again model = model.eval()
    # TODO test with this again model = torch.compile(model)

log("LLaMA ready.")

# 5 is the number of paraphrases to generate 
# 1.65 states that the paraphrase can be 65% longer than the input sentence
PROMPT_LENGTH_MULTIPLIER = 1.65 * 5 

PROMPT = {
    "en": """[INST] <<SYS>>
You are a professional writer, your expertise is in crafting accurate and engaging paraphrases.
<</SYS>>
Generate five distinct English paraphrases of the sentence provided below. Each paraphrase should fully convey the original meaning without adding extraneous information. Aim for a balance between retaining the essence of the sentence and presenting it in a fresh, clear manner.

Original Sentence: 'We know that, right? We've experienced that.'
[/INST]Paraphrases:
1. "That's something we understand, isn't it? It's been part of our experiences."
2. "We're aware of that, correct? We've gone through it ourselves."
3. "That's known to us, right? We have lived through that."
4. "We are cognizant of that, aren't we? We've felt that in our own lives."
5. "Isn't that familiar to us? We've personally encountered it."
</s><s>[INST]
Original Sentence: '{}'
[/INST]Paraphrases:
1.""",
    "de": """[INST] <<SYS>>
Sie sind ein professioneller deutscher Schriftsteller, der sich auf das Verfassen von Paraphrasen spezialisiert hat.
<</SYS>>
Erstellen sie fünf verschiedene Paraphrasen des untenstehenden Satzes in flüssigem und natürlichem Deutsch. Jede Paraphrase sollte die ursprüngliche Bedeutung vollständig vermitteln, ohne zusätzliche Informationen hinzuzufügen. Achten Sie darauf, dass ausschließlich Deutsch verwendet wird und der Satzbau klar und korrekt ist.

Originalsatz: 'Um das zu verdeutlichen, habe ich mir ein kleines Spiel ausgedacht.'
[/INST]Deutsche Paraphrasen:
1. "Zur Veranschaulichung habe ich ein kleines Spiel entwickelt."
2. "Ein kleines Spiel soll das besser illustrieren, das ich mir ausgedacht habe."
3. "Ich habe ein kleines Spiel konzipiert, um dies zu demonstrieren."
4. "Um das klarzumachen, entwarf ich ein kleines Spiel."
5. "Ein kleines Spiel, das ich erfunden habe, soll dies verdeutlichen."
</s><s>[INST]
Originalsatz: '{}'
[/INST]Deutsche Paraphrasen:
1."""
}

BASE_PROMPT_TOKEN_LENGTH = {
    "en": len(TOKENIZER["en"].encode(PROMPT["en"].format(""))) - 1,
    "de": len(TOKENIZER["de"].encode(PROMPT["de"].format(""))) - 1,
}

log(f"Base prompt token length: {BASE_PROMPT_TOKEN_LENGTH}")

PARAPHRASE_GENERATE_TIME = 0 # 60 * 60 * 12 # 12 hours # TODO temporarily disabled

def generate(prompts: list[str], lng: LANGUAGE) -> list[str]:
    model_inputs = TOKENIZER[lng](prompts, return_tensors="pt", padding=True).to(DEVICE)
    
    max_input_length = model_inputs.input_ids.shape[-1]
    max_new_tokens = BASE_PROMPT_TOKEN_LENGTH[lng] + (max_input_length - BASE_PROMPT_TOKEN_LENGTH[lng]) * PROMPT_LENGTH_MULTIPLIER
 
    generation_config = GenerationConfig(
        temperature=0.4, # TODO really low temperature but still get hallucinations
        top_p=0.75,
        repetition_penalty=1.1,
        do_sample=True,
        num_beams=1,
    )

    with torch.inference_mode():
        generated_ids = LLM[lng].generate(
            **model_inputs,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
        )
    
    decoded_outputs = TOKENIZER[lng].batch_decode(generated_ids, skip_special_tokens=True)
    decoded_inputs = TOKENIZER[lng].batch_decode(model_inputs.input_ids, skip_special_tokens=True)
    
    return [
        output.replace(prompt, "").strip()
        for output, prompt in zip(decoded_outputs, decoded_inputs)
    ]

# Generates 5 paraphrases for the input sentence with LLaMA 2
def generate_batched_paraphrases(sentences: list[str], language: LANGUAGE) -> list[list[str]]:    
    log(f"\nGenerating paraphrases in '{language}'...")
    
    sentence_too_long = [len(sentence) > 100 for sentence in sentences]
    short_sentences = [sentence for sentence, too_long in zip(sentences, sentence_too_long) if not too_long]

    # Format the prompt with the given sentence
    prompted = [PROMPT[language].format(sentence) for sentence in short_sentences]

    # Generate response using LLaMA 2
    paraphrased_texts = generate(prompted, language)
    
    all_paraphrases = [[sentence] for sentence in sentences]
    
    paraphrased_texts_index = 0
    for i, sentence in enumerate(sentences):
        if sentence_too_long[i]:
            continue
    
        log(f"\n\nParaphrases text for sentence '{sentence}':")
        #log(f"\"\"\"{paraphrases_text}\"\"\"")
        
        paraphrased_text = paraphrased_texts[paraphrased_texts_index]
        paraphrased_texts_index += 1

        sentences = list(set((
            cleanup_paraphrase(sent) for sent in paraphrased_text.split("\n") if cleanup_paraphrase(sent)
        )))

        # Heuristics to identify and extract paraphrases
        paraphrases = [
            sent for sent in sentences if heuristic_is_paraphrase(sent, sentence, language)
        ]
        non_paraphrases = [
            sent for sent in sentences if sent not in paraphrases
        ]
        log("Non paraphrases:")
        for sent in non_paraphrases:
            log(f"    {sent}")

        if len(paraphrases) < 5:
            log(f"\nWarning: Only {len(paraphrases)} paraphrases generated for '{sentence}'\n")
            
        log("Paraphrases:")
        for sent in paraphrases:
            log(f"    {sent}")

        # Ensure only five paraphrases are returned    
        all_paraphrases[i].extend(paraphrases) # TODO currently return all [:5]

    return all_paraphrases 


def cleanup_paraphrase(sent: str) -> str:
    sent = sent.replace("<s>", "").replace("</s>", "").replace("<unk>", "").strip()
    
    # if it ends with a ')' then replace the last bracketed part with ''
    if sent.endswith(")"):
        # find the first matching bracket from the end of the sentence
        open_brackets = 0
        for i, char in enumerate(reversed(sent)):
            if char == ")":
                open_brackets += 1
            elif char == "(":
                open_brackets -= 1
                
            if open_brackets == 0:
                # remove the bracketed part
                sent = sent[:-i]
                break            
    
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
    # WMT 14-19 datasets, but generate a set from these to avoid duplicates
    s = set()    
    total_sizes = 0
    
    for i in range(14, 20):
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
    log(f"Total dataset length: {len(s)}")
    log(f"Removed {total_sizes - len(s)} duplicates.")
    
    return s

def our_data_generator():    
    def ensure_dataset_loaded() -> None:
        if not DATASET_EN.is_file() or not DATASET_DE.is_file():

            download_location = DATASET_FOLDER / "data.zip"
            # Download and unzip command
            download_command = f"wget -nc -O '{download_location}' '{DATASET_URL}'"
            unzip_command = f"unzip -o '{download_location}' -d '{DATASET_FOLDER}'"

            # Execute download and unzip commands
            log("Checking and downloading dataset...")
            os.system(download_command)
            log("Unzipping dataset...")
            os.system(unzip_command)
            log("Dataset ready.")

    ensure_dataset_loaded()
    
    # Read the dataset segment for this process
    english_sentences = read_dataset_segment(DATASET_EN)
    german_sentences = read_dataset_segment(DATASET_DE)
     
    if len(english_sentences) != len(german_sentences):
        log("Warning: English and German sentences are not the same length.")
    
    log(f"Original dataset length: {len(english_sentences)}")
    
    yield from zip(english_sentences, german_sentences)
    
     
def batch_tuples(iterable, batch_size=1, values_per_tuple=2):
    # take in a generator of tuples and yield a generator of tuples of batch_size
    # i.e. [(1, 2), (3, 4), (5, 6)] should yield [((1,3), (2,4)), ((5,), (6,))] for batch_size=2
    
    items = []
    for item in iterable:
        items.append(item)
        if len(items) == batch_size:
            lists = [[] for _ in range(values_per_tuple)]
            for tup in items:
                for i, item in enumerate(tup):
                    lists[i].append(item)
            yield tuple(lists)            
            items = []
    
    lists = [[] for _ in range(values_per_tuple)]
    for tup in items:
        for i, item in enumerate(tup):
            lists[i].append(item)
    yield tuple(lists)
                
def main() -> None:
    log("Generating paraphrases for all sentence pairs...")
    
    if PARAPHRASE_GENERATE_TIME != 0:
        with open(OUTPUT_EN_FILE, "w", encoding="utf-8") as en_file,\
            open(OUTPUT_DE_FILE, "w", encoding="utf-8") as de_file, \
            open(LOG_FILE, "a", encoding="utf-8") as log_file:
                
            start_paraphrasing = time.time()
            total_paraphrases = 0
            total_written_paraphrases = 0
                
            for ens, des in batch_tuples(our_data_generator(), 10): # TODO experiment, are larger batches better?
                if time.time() - start_paraphrasing > PARAPHRASE_GENERATE_TIME:
                    log("12 hours have passed. Stopping paraphrasing.")
                    break
                
                start = time.time()
                log(f"\n\nGenerating paraphrases for '{ens}' and '{des}'...")
                try:
                    en_paraphrases = generate_batched_paraphrases(ens, "en")
                    de_paraphrases = generate_batched_paraphrases(des, "de")
                    log(f"\nParaphrases generated in {round(time.time() - start, 2)} seconds.")
                    new_paraphrases = sum(len(paraphrases) for paraphrases in en_paraphrases) + sum(len(paraphrases) for paraphrases in de_paraphrases) - len(ens) - len(des)
                    log(f"New paraphrases generated: {new_paraphrases}")
                    total_paraphrases += new_paraphrases
                except Exception as e:
                    log(f"Error generating paraphrases: {e}")
                    continue

                # Generate all combinations of English and German paraphrases and write directly to files
                for en_ps, de_ps in zip(en_paraphrases, de_paraphrases):
                    for en_p, de_p in itertools.product(en_ps, de_ps):
                        en_file.write(f"{en_p}\n")
                        de_file.write(f"{de_p}\n")
                        en_file.flush()
                        de_file.flush()
                        total_written_paraphrases += 1
                    
                    # Json dump the paraphrases to the log file
                    json.dump({
                        "en_paraphrases": en_ps,
                        "de_paraphrases": de_ps,
                    }, log_file)
                    log_file.write("\n")
                    log_file.flush()
                    
                log(f"Total paraphrases generated: {total_paraphrases}")
                log(f"Total paraphrases written: {total_written_paraphrases}")
                log(f"Total time taken: {round(time.time() - start_paraphrasing, 2)} seconds")

        log("Finished generating paraphrases.")

        spm.SentencePieceTrainer.train(input=f"{OUTPUT_DE_FILE},{OUTPUT_EN_FILE}",
                                    model_prefix="bpe",
                                    vocab_size=5000)

        log('Finished training sentencepiece model.')
    
    spm_model = spm.SentencePieceProcessor(model_file="bpe.model")
    
    log("BPE model ready.")
    
    datasets = data_generator()

    for file, lang in zip((OUTPUT_DE_FILE, OUTPUT_EN_FILE), ("de", "en")):
        log(f"Segmenting {lang} dataset...")
        with open(file, "r", encoding="utf-8") as f_in, \
                open(SPM_OUTPUT_FILE.as_posix().format(lang), "w", encoding="utf-8") as f_out:
            for line in tqdm(f_in.readlines(), desc=f"Segmenting {lang} our Dataset"):
                # Segmented into subwords
                line_segmented = spm_model.encode(line.strip(), out_type=str)
                # Join the subwords into a string
                line_segmented = " ".join(line_segmented)
                f_out.write(line_segmented + "\n")
                
            f_out.flush()
            
            for en, de in tqdm(datasets, desc=f"Segmenting {lang} WMT"):
                # Segmented into subwords
                line = en if lang == "en" else de
                line_segmented = spm_model.encode(line.strip(), out_type=str)
                # Join the subwords into a string
                line_segmented = " ".join(line_segmented)
                f_out.write(line_segmented + "\n")

            f_out.flush()

if __name__ == "__main__":
    main()

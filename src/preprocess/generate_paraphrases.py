import os
import json
import time
import regex
import spacy
import itertools

from typing import Literal
from torch.utils.data import DataLoader
from src.datasets.concrete.covost import CoVoSTWithText

from src.llama.llama import generate
from src.logger_utils import get_logger
from src.paths import COVOST_MT_PARAPHRASED_DE_FILE, COVOST_MT_PARAPHRASED_EN_FILE, COVOST_ROOT, MT_COVOST_DATA_ROOT

logger = get_logger("Paraphrases::generate paraprhases")

LOG_FILE = MT_COVOST_DATA_ROOT / "paraphrases.log"

TOTAL_PROCESSING_TIME_IN_HOURS = int(os.getenv("TOTAL_PROCESSING_TIME_IN_HOURS", 24))

LANGUAGE = Literal["en"] | Literal["de"]

try:
    NLP = {"en": spacy.load("en_core_web_sm"), "de": spacy.load("de_core_news_sm")}
except OSError:
    logger.info("Downloading Spacy models...")
    os.system("python -m spacy download en")
    os.system("python -m spacy download de")
    NLP = {"en": spacy.load("en_core_web_sm"), "de": spacy.load("de_core_news_sm")}
    
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

def generate_batched_paraphrases(sentences: list[str], language: LANGUAGE) -> list[list[str]]:    
    logger.info(f"\nGenerating paraphrases in '{language}'...")
    
    sentence_too_long = [len(sentence) > 100 for sentence in sentences]
    short_sentences = [sentence for sentence, too_long in zip(sentences, sentence_too_long) if not too_long]

    # Format the prompt with the given sentence
    prompted = [PROMPT[language].format(sentence) for sentence in short_sentences]

    # Generate response using LLaMA 2
    paraphrased_texts = generate(prompted)
    
    all_paraphrases = [[sentence] for sentence in sentences]
    
    paraphrased_texts_index = 0
    for i, sentence in enumerate(sentences):
        if sentence_too_long[i]:
            continue
    
        logger.info(f"\n\nParaphrases text for sentence '{sentence}':")
        
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
        logger.info("Non paraphrases:")
        for sent in non_paraphrases:
            logger.info(f"    {sent}")

        if len(paraphrases) < 5:
            logger.info(f"\nWarning: Only {len(paraphrases)} paraphrases generated for '{sentence}'\n")
            
        logger.info("Paraphrases:")
        for sent in paraphrases:
            logger.info(f"    {sent}")

        # Ensure only five paraphrases are returned    
        all_paraphrases[i].extend(paraphrases)

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

                
if __name__ == "__main__":
    logger.info("Generating paraphrases for all sentence pairs...")
    
    with open(COVOST_MT_PARAPHRASED_EN_FILE, "a", encoding="utf-8") as en_file,\
        open(COVOST_MT_PARAPHRASED_DE_FILE, "a", encoding="utf-8") as de_file, \
        open(LOG_FILE, "a", encoding="utf-8") as log_file:
            
        start_paraphrasing = time.time()
        total_paraphrases = 0
        total_written_paraphrases = 0
            
        dataset = CoVoSTWithText(COVOST_ROOT, "train", "en", "de")
        
        for ens, des in DataLoader(dataset, batch_size=20): # TODO experiment, are larger batches better?
            logger.info(f"\n\nGenerating paraphrases for '{ens}' and '{des}'...")
            try:
                start = time.time()
                
                en_paraphrases = generate_batched_paraphrases(ens, "en")
                de_paraphrases = generate_batched_paraphrases(des, "de")
                
                logger.info(f"\nParaphrases generated in {round(time.time() - start, 2)} seconds.")
                
                new_paraphrases = sum(len(paraphrases) for paraphrases in en_paraphrases) + \
                                    sum(len(paraphrases) for paraphrases in de_paraphrases) - \
                                    len(ens) - len(des)
                                    
                logger.info(f"New paraphrases generated: {new_paraphrases}")
                total_paraphrases += new_paraphrases
            except Exception as e:
                logger.info(f"Error generating paraphrases: {e}")
                continue

            # Generate all combinations of English and German paraphrases and write directly to files
            for en_ps, de_ps in zip(en_paraphrases, de_paraphrases):
                for en_p, de_p in itertools.product(en_ps, de_ps):
                    en_file.write(f"{en_p}\n")
                    de_file.write(f"{de_p}\n")
                    total_written_paraphrases += 1
                
                # Json dump the paraphrases to the log file
                json.dump({
                    "en_paraphrases": en_ps,
                    "de_paraphrases": de_ps,
                }, log_file)
                log_file.write("\n")
                
            en_file.flush()
            de_file.flush()
            log_file.flush()
                
            logger.info(f"Total paraphrases generated: {total_paraphrases}")
            logger.info(f"Total paraphrases written: {total_written_paraphrases}")
            logger.info(f"Total time taken: {round(time.time() - start_paraphrasing, 2)} seconds")
            
            if (time.time() - start_paraphrasing) / 3600 > TOTAL_PROCESSING_TIME_IN_HOURS:
                logger.info(f"Total processing time exceeded {TOTAL_PROCESSING_TIME_IN_HOURS} hours. Exiting.")
                break

    logger.info("Finished generating paraphrases.")

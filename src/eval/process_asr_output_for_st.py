import argparse
import os
import subprocess
from tqdm import tqdm

from src.mt.bpe import BPE
from src.asr.config import cleanup_utterance
from src.logger_utils import get_logger
from src.datasets.concrete.covost import CoVoST, CoVoSTWithText

from src.paths import COVOST_ROOT, MT_SPM_MODEL, PUNCTUATION_SPM_MODEL, MT_ROOT


logger = get_logger("Eval::process_asr_output_for_st")

parser = argparse.ArgumentParser()
parser.add_argument("--ref_input_file", type=str)
parser.add_argument("--ref_output_file", type=str)
parser.add_argument("--hyp_input_file", type=str)
parser.add_argument("--hyp_output_file", type=str)
parser.add_argument("--type_of_postprocessing", type=str, default="llama") # "none", "llama", "custom"
parser.add_argument("--num_samples_per_prediction", type=int, default=10)
parser.add_argument("--num_samples_to_evaluate", type=int, default=100)
parser.add_argument("--src_lng", type=str, default="en")
parser.add_argument("--target_lng", type=str, default="de")
args = parser.parse_args()


def no_postprocessing(lines: list[str]) -> list[str]:
    return lines

def llama_postprocessing(lines: list[str]) -> list[str]:
    from src.llama.llama import generate
    
    LLM_POST_EDITING_PROMPT = """[INST] <<SYS>>
You are a professional specialized in ASR (Automatic Speech Recognition) transcription enhancement.
<</SYS>>
Task:
1. Punctuation Restoration: Add necessary punctuation to make the sentences grammatically correct and more readable.
2. Evaluate the hypotheses for accuracy and likelihood of being the correct representation of the spoken text.
3. Select the best hypothesis that accurately reflects the original spoken text.

Input list of the 10 best ASR hypotheses:
1. "mr jones said meet me at 10 am in the conference room"
2. "mister jones said meet me at ten a m in the conference room"
3. "mr jones said meet me at ten am in the conference room"
4. "mister jones said meet me at 10 am in the conference room"
5. "mr jones said meet me at ten in the morning in the conference room"
6. "mr jones said meet me at 10 in the conference room"
7. "mister jones said meet me at ten in the conference room"
8. "mr jones said meet me at 10 in the conference room"
9. "mr jones said meet me at ten a m in the conference room"
10. "mister jones said meet me at 10 a m in the conference room"[/INST]
Best Hypothesis: "Mr. Jones said, 'Meet me at 10 a.m. in the conference room.'"

</s><s>[INST]
Input list of the 10 best ASR hypotheses:
{HYPOTHESES}[/INST]
Best Hypothesis: \""""
    
    processed_lines = []

    # process in batches of 10
    batch_size = 10
    batch = []
    for i in tqdm(range(0, len(lines), args.num_samples_per_prediction), desc="Processing batches"):
        samples = lines[i:i + args.num_samples_per_prediction]

        formatted_samples = "\n".join(f'{i + 1}. "{sample}"' for i, sample in enumerate(samples))

        prompt = LLM_POST_EDITING_PROMPT.format(HYPOTHESES=formatted_samples)
        batch.append(prompt)

        if len(batch) == batch_size:
            processed_lines.extend(generate(batch))
            batch = []

    if len(batch) > 0:
        processed_lines.extend(generate(batch))

    def cleanup_output(output: str) -> str:
        output = output.split("\n")[0].strip() # only get the first line of the output
        last = output
        while last != output:
            last = output
            output = output.replace("  ", " ") # remove double spaces
            output = output.replace(" ,", ", ") # remove spaces before commas

        # the output should always be in 'Best Hypothesis: "..."' format so we can just extract the text in between the quotes
        output = output.split('"')[0]

        return output

    return [cleanup_output(line) for line in processed_lines]

def custom_postprocessing(lines: list[str]) -> list[str]:
    TRAIN_WORKSPACE=MT_ROOT / "train" / "train_punctuation_covost"
    BINARY_DATA_DIR=TRAIN_WORKSPACE / "binarized_dataset"
    MODEL_DIR=TRAIN_WORKSPACE / "models"
    PRECONDITIONS_DIR = f"{os.environ['HOME']}/preconditions/eval_st/punctuation"
        
    bpe = BPE.from_pretrained(PUNCTUATION_SPM_MODEL)
    
    encoded_lines = bpe.encode_lines(lines)
    
    punctuation_spm_file = args.hyp_input_file + ".punctuation.spm.en"
    with open(punctuation_spm_file, "w", encoding="utf-8") as f:
        f.write("\n".join(encoded_lines))
        
    # call generate on the file
    TEST_PREF = punctuation_spm_file.replace(".en", "")
    COMMAND = f"./src/bash/translate_mt.sh {BINARY_DATA_DIR}/dict.en.txt {BINARY_DATA_DIR}/dict.de.txt {TEST_PREF} {BINARY_DATA_DIR.as_posix()} {MODEL_DIR.as_posix()} {PRECONDITIONS_DIR}"
    subprocess.run([COMMAND], shell=True)
    
    # read the predictions
    with open(PRECONDITIONS_DIR + "/hyp_mt.txt", "r", encoding="utf-8") as f:
        predictions = [line.strip() for line in f.readlines()]
        
    # decode the predictions with bpe
    decoded_predictions = bpe.decode_lines(predictions)
    
    return decoded_predictions    


TYPES_OF_POSTPROCESSING = {
    "none": no_postprocessing,
    "llama": llama_postprocessing,
    "custom": custom_postprocessing,
}


def __sample_print(data_list: list[str]) -> None:
    logger.info("----------------------------------------")
    for data in data_list[:2]:
        logger.info(data)
    logger.info("----------------------------------------")


def __encode_and_save(lines: list[str], output_file: str) -> list[str]:
    bpe = BPE.from_pretrained(MT_SPM_MODEL)
    
    encoded_lines = bpe.encode_lines(lines)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(encoded_lines))
    
    return encoded_lines


def process_reference_file() -> None:
    with open(args.ref_input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    lines_subsample = lines[:args.num_samples_to_evaluate]
    lines_processed = process_reference_text(lines_subsample)

    logger.info("Processed reference file")
    __sample_print(lines_processed)

    __encode_and_save(lines_processed, args.ref_output_file)

    logger.info("Done processing reference file!")


def process_hypothesis_file() -> None:
    logger.info("Processing hypothesis file...")
    with open(args.hyp_input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    lines_subsample = lines[:args.num_samples_to_evaluate*args.num_samples_per_prediction]
    lines_processed = process_hypothesis_text(lines_subsample)

    logger.info("Processed hypothesis file")
    __sample_print(lines_processed)

    __encode_and_save(lines_processed, args.hyp_output_file)

    logger.info("Done processing hypothesis file!")


def process_reference_text(lines: list[str]) -> list[str]:
    covost = CoVoSTWithText(COVOST_ROOT, CoVoST.SPLITS[2], "en", "de")
    
    translation_dict = {
        cleanup_utterance(original): target
        for original, target in covost
    }
    
    logger.info("Translating file...")

    translations = [
        translation_dict[line] if line in translation_dict else line.lower().strip()
        for line in lines
    ]

    __sample_print(translations)

    return translations


def process_hypothesis_text(lines: list[str]) -> list[str]:
    """Post-processes the hypothesis text. For example, add punctuation, capitalization, etc. Returns a list of post-processed hypothesis texts in the same order as the input lines."""
    return TYPES_OF_POSTPROCESSING[args.type_of_postprocessing](lines)


if __name__ == "__main__":
    logger.info("Starting processing...")

    process_hypothesis_file()

    process_reference_file()

    logger.info("Done processing!")

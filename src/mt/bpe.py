from pathlib import Path
import sentencepiece as spm

from tqdm import tqdm

from src.logger_utils import get_logger

logger = get_logger("MachineTranslation::BPE")


def process_lines(file_path: Path, encoding="utf-8") -> list[str]:
    buffer_size = 4096  # Size of the chunk to read
    partial_line = b""  # To store a partial line at the end of a buffer
    processed_lines = []  # List to store processed lines

    with open(file_path, "rb") as file:
        while True:
            buf = partial_line + file.read(buffer_size)
            if not buf:  # End of file
                break

            buffer_lines = buf.split(b"\n")
            partial_line = buffer_lines.pop()  # Handle the last partial line

            for line in buffer_lines:
                decoded_line = line.decode(encoding, errors="ignore")
                processed_line = decoded_line.replace("\n", "").replace("\r", "")
                processed_lines.append(processed_line)

    # Check if there is any remaining part after the last newline
    if partial_line:
        decoded_line = partial_line.decode(encoding, errors="ignore")
        processed_line = decoded_line.replace("\n", "").replace("\r", "")
        processed_lines.append(processed_line)

    return processed_lines


class BPE:
    def __init__(self, retrain_spm: bool, train_files: list[str]|None=None, vocab_size=10000, model_prefix="bpe") -> None:
        if retrain_spm:
            assert train_files is not None
            logger.info("Retraining sentencepiece model...")

            spm.SentencePieceTrainer.train( # type: ignore
                input=", ".join(train_files),
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                input_sentence_size=1000000,
                shuffle_input_sentence=True,
            )

            logger.info("Finished training sentencepiece model.")

        self.spm_model = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model") # type: ignore

    def encode_file(self, input_file: Path, output_file: Path, overwrite:bool=False) -> None:
        self.__process(input_file, output_file, overwrite, process_fn=self.__encode, name="Encoding")

    def decode_file(self, input_file: Path, output_file: Path, overwrite:bool=False) -> None:
        self.__process(input_file, output_file, overwrite, process_fn=self.__decode, name="Decoding")
                
    def __process(self, input_file: Path, output_file: Path, overwrite: bool, process_fn, name) -> None:
        if output_file.is_file() and not overwrite:
            logger.info(f"Skipping {name} of {input_file} because {output_file} already exists.")
            return
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f_out:
            for line in tqdm(process_lines(input_file), desc=f"{name} '{input_file.name}' Dataset"):
                # Process line
                line_processed = process_fn(line)
                f_out.write(" ".join(line_processed) + "\n")

    def __encode(self, text: str) -> list[str]:
        return self.spm_model.encode(text.strip(), out_type=str) # type: ignore

    def __decode(self, text: str) -> list[str]:
        return self.spm_model.decode(text.strip(), out_type=str) # type: ignore


if __name__ == "__main__":
    data = Path("data")

    bpe = BPE(
        retrain_spm=True,
        train_files=["data/train.en", "data/train.de"],
        vocab_size=15000,
    )

    bpe.encode_file(data / "train.en", data / "spm.train.en")
    bpe.decode_file(data / "spm.train.en", data / "train.en")

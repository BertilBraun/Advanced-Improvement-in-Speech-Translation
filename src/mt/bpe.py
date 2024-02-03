from pathlib import Path
from typing import Callable
import sentencepiece as spm

from tqdm import tqdm

from src.logger_utils import get_logger

logger = get_logger("MachineTranslation::BPE")


def read_in_lines(file_path: Path, encoding="utf-8") -> list[str]:
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

def write_lines(lines: list[str], output_file: Path | str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class BPE:
    @staticmethod
    def from_pretrained(model_file: Path) -> "BPE":
        return BPE(retrain_spm=False, model_file=model_file)
    
    @staticmethod
    def write_encoded_lines(model_file: Path, lines: list[str], output_file: Path | str) -> list[str]:
        bpe = BPE.from_pretrained(model_file)
        encoded_lines = bpe.encode_lines(lines)
        write_lines(encoded_lines, output_file)
        return encoded_lines
    
    def __init__(self, retrain_spm: bool, model_file: Path, train_files: list[str]|None=None, vocab_size=10000) -> None:
        if retrain_spm or not model_file.is_file():
            assert train_files is not None
            assert model_file.suffix == ".model"
            logger.info("Retraining sentencepiece model...")

            spm.SentencePieceTrainer.Train( 
                input=", ".join(train_files),
                model_prefix=model_file.stem,
                vocab_size=vocab_size,
                input_sentence_size=1000000,
                shuffle_input_sentence=True,
            )

            logger.info("Finished training sentencepiece model.")

        self.spm_model = spm.SentencePieceProcessor()
        self.spm_model.Init(model_file=model_file.as_posix()) 

    def encode_file(self, input_file: Path, output_file: Path, overwrite:bool=False) -> None:
        self._process(input_file, output_file, overwrite, process_fn=self.encode_lines, name="Encoding")

    def decode_file(self, input_file: Path, output_file: Path, overwrite:bool=False) -> None:
        self._process(input_file, output_file, overwrite, process_fn=self.decode_lines, name="Decoding")
        
    def encode_lines(self, lines: list[str]) -> list[str]:
        return [
            " ".join(self._encode(line.strip()))
            for line in tqdm(lines, desc="Encoding Dataset")
        ]
    
    def decode_lines(self, lines: list[str]) -> list[str]:
        return [
            self._decode(line.strip().split(" "))
            for line in tqdm(lines, desc="Decoding Dataset")
        ]
                        
    def _process(self, input_file: Path, output_file: Path, overwrite: bool, process_fn: Callable[[list[str]], list[str]], name: str) -> None:
        if output_file.is_file() and not overwrite:
            logger.info(f"Skipping {name} of {input_file} because {output_file} already exists.")
            return
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        input_lines = read_in_lines(input_file)
        processed_lines = process_fn(input_lines)
        write_lines(processed_lines, output_file)
            
    def _encode(self, text: str) -> list[str]:
        return self.spm_model.Encode(text.strip(), out_type=str)

    def _decode(self, text: list[str]) -> str:
        return self.spm_model.Decode(text, out_type=str)


if __name__ == "__main__":
    data = Path("data")

    bpe = BPE(
        retrain_spm=True,
        model_file=data / "spm.model",
        train_files=["data/train.en", "data/train.de"],
        vocab_size=15000,
    )

    bpe.encode_file(data / "train.en", data / "spm.train.en")
    bpe.decode_file(data / "spm.train.en", data / "train.en")

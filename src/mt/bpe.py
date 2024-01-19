import sentencepiece as spm

from tqdm import tqdm


def process_lines(file_path: str, encoding="utf-8") -> list[str]:
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
    def __init__(
        self,
        retrain_spm: bool,
        train_files: list[str],
        vocab_size=10000,
        model_prefix="bpe",
    ) -> None:
        if retrain_spm:
            print("Retraining sentencepiece model...")

            spm.SentencePieceTrainer.train(
                input=", ".join(train_files),
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                input_sentence_size=1000000,
                shuffle_input_sentence=True,
            )

            print("Finished training sentencepiece model.")

        self.spm_model = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

    def encode_file(self, input_file: str, output_file: str) -> None:
        with open(output_file, "w", encoding="utf-8") as f_out:
            file_name = input_file.split("/")[-1]
            for line in tqdm(
                process_lines(input_file), desc=f"Segmenting '{file_name}' Dataset"
            ):
                # Segmented into subwords
                line_segmented = self.__encode(line)
                f_out.write(" ".join(line_segmented) + "\n")

    def __encode(self, text: str) -> list[str]:
        return self.spm_model.encode(text.strip(), out_type=str)

    def decode_file(self, input_file: str, output_file: str) -> None:
        with open(output_file, "w", encoding="utf-8") as f_out:
            file_name = input_file.split("/")[-1]
            for line in tqdm(
                process_lines(input_file), desc=f"Desegmenting '{file_name}' Dataset"
            ):
                # Desegment into words
                line_desegmented = self.__decode(line)
                f_out.write(" ".join(line_desegmented) + "\n")

    def __decode(self, text: str) -> list[str]:
        return self.spm_model.decode(text.strip(), out_type=str)


if __name__ == "__main__":
    bpe = BPE(
        retrain_spm=True,
        train_files=["data/train.en", "data/train.de"],
        vocab_size=15000,
    )

    bpe.encode_file("data/train.en", "data/spm.train.en")

    bpe.decode_file("data/spm.train.en", "data/train.en")

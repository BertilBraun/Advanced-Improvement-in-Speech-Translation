import os
import numpy as np
import torch
import shutil
import datetime
import sentencepiece as spm
from evaluate import load
from datasets import load_dataset, Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Speech2TextForConditionalGeneration, Speech2TextConfig, TrainingArguments, Trainer, Speech2TextTokenizer, DataCollatorWithPadding

wer_metric = load("wer")

TRAIN_DATASET = "train.100"

TOKENIZER_PATH = "tokenizer.model"
TOKENIZER_VOCAB_SIZE = 900

INPUT_VALUES = "input_values"
LABELS = "labels"


def train_and_save_tokenizer():
    if not os.path.isfile(TOKENIZER_PATH):
        print("Training tokenizer...")
        TOKENIZER_VOCAB_TRAIN_FILE = "librispeech_train_text.txt"
        
        dataset = load_dataset("librispeech_asr", "clean", split=TRAIN_DATASET)
        
        with open(TOKENIZER_VOCAB_TRAIN_FILE, 'w', encoding="utf-8") as f:
            for item in dataset:
                f.write(item['text'] + '\n')
        
        spm.SentencePieceTrainer.train(
            input=TOKENIZER_VOCAB_TRAIN_FILE,
            vocab_size=TOKENIZER_VOCAB_SIZE,
            model_type="bpe",
            model_prefix=TOKENIZER_PATH.split(".")[0],
        )
        
        print("Tokenizer trained")
    
    return spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    

tokenizer = train_and_save_tokenizer()
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


def compute_metrics(pred):
    predictions = pred.predictions
    references = pred.label_ids

    # Flatten the batch dimension if necessary
    if predictions.ndim == 3:  # (batch_size, seq_len, vocab_size)
        predictions = np.argmax(predictions, axis=-1)

    # Decode predictions and references
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(references, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    decoded_labels = [label if label != -100 else "" for label in decoded_labels]

    # Compute WER
    return {"wer": wer_metric.compute(predictions=decoded_preds, references=decoded_labels)}

# Function to automatically find the latest checkpoint
def find_latest_checkpoint(output_dir):
    checkpoints = [os.path.join(output_dir, name) for name in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, name))]
    if checkpoints:
        return max(checkpoints, key=os.path.getmtime)
    return None

def prepare_data(batch):
    inputs = processor(batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"], return_tensors="pt").input_values
    with torch.no_grad():
        embeddings = wav2vec_model(inputs).last_hidden_state.squeeze()

    # Tokenize the labels (transcriptions)
    labels = tokenizer.encode(batch["text"], out_type=int)

    # Update the batch
    batch[INPUT_VALUES] = embeddings
    batch[LABELS] = torch.tensor(labels)

    return batch

def padd_data(percentile_95_audio, percentile_95_labels, feature_size=768):
    def _padd_data(batch):
        inputs = batch[INPUT_VALUES]  # List of 768-dim tensors
        labels = batch[LABELS]        # List of ints

        # Convert each input to a tensor if they are not already
        inputs = [torch.tensor(input, dtype=torch.float) if not isinstance(input, torch.Tensor) else input for input in inputs]

        # Stack the inputs to create a single tensor
        inputs = torch.stack(inputs)

        # Initialize padded tensors
        padded_inputs = torch.zeros((percentile_95_audio, feature_size), dtype=torch.float)
        padded_labels = torch.full((percentile_95_labels,), tokenizer.pad_id(), dtype=torch.long)

        # Calculate actual lengths and ensure they don't exceed the predetermined padding sizes
        actual_seq_len = min(inputs.shape[0], percentile_95_audio)
        actual_label_len = min(len(labels), percentile_95_labels)

        # Copy data to padded tensors
        padded_inputs[:actual_seq_len, :] = inputs[:actual_seq_len, :]
        padded_labels[:actual_label_len] = torch.tensor(labels, dtype=torch.long)[:actual_label_len]

        # Update the batch
        batch[INPUT_VALUES] = padded_inputs
        batch[LABELS] = padded_labels

        return batch
    return _padd_data

def calculate_percentiles(processed_dataset):
    # Calculate lengths
    audio_lengths = [len(data[INPUT_VALUES]) for data in processed_dataset]
    label_lengths = [len(data[LABELS]) for data in processed_dataset]

    # Calculate 95th percentiles
    percentile_95_audio = np.percentile(audio_lengths, 95)
    percentile_95_labels = np.percentile(label_lengths, 95)

    return int(percentile_95_audio), int(percentile_95_labels)

def load_and_preprocess_data(split, dataset_path):
    dataset_file = f"{dataset_path}/{split}_processed.pt"
    if os.path.exists(dataset_file):
        print(f"Loading processed {split} dataset from {dataset_file}")
        return Dataset.load_from_disk(dataset_file)
    else:
        print(f"Loading and processing {split} dataset...")
        dataset = load_dataset("librispeech_asr", "clean", split=split)

        # TODO Optional: Subset the dataset if needed
        dataset = dataset.select(range(20))  # Select first 50 samples for testing

        processed_dataset = dataset.map(prepare_data, remove_columns=["audio"])

        print("Calculating 95th percentiles...")        
        percentile_95_audio, percentile_95_labels = calculate_percentiles(processed_dataset)
        print(f"95th percentile audio length: {percentile_95_audio}")
        print(f"95th percentile label length: {percentile_95_labels}")
        print("Padding data...")        
        padded_dataset = processed_dataset.map(padd_data(percentile_95_audio, percentile_95_labels))
       
        padded_dataset.set_format(type="torch", columns=[INPUT_VALUES, LABELS])
        print(f"Processed {split} dataset, saving to {dataset_file}...")
        
        padded_dataset.save_to_disk(dataset_file)
        return padded_dataset

dataset_path = "./processed_datasets"
os.makedirs(dataset_path, exist_ok=True)

train_ds = load_and_preprocess_data(TRAIN_DATASET, dataset_path)
val_ds = load_and_preprocess_data("validation", dataset_path)
test_ds = load_and_preprocess_data("test", dataset_path)

print("Successfully loaded all datasets")

print(train_ds[0].keys())

# Configure and initialize the S2T model
config = Speech2TextConfig(
    vocab_size=900,
    # Adjusted to match wav2vec's output
    input_feat_per_channel=768,
)
model = Speech2TextForConditionalGeneration(config)

# "YYYY-MM-DD HH:MM"
date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

# Training arguments
output_dir = f"results/{date_time}/"  # Directory to save checkpoints to
log_dir = f"logs/{date_time}/"  # Directory to save logs to

output_dir = "results"
log_dir = "logs"

# delete everything in the output directory and log directory and the directories themselves
shutil.rmtree(output_dir, ignore_errors=True)
shutil.rmtree(log_dir, ignore_errors=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=150,
    per_device_train_batch_size=16, 
    gradient_accumulation_steps=8, 
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir=log_dir,
    logging_steps=10,
    learning_rate=2e-3,
    max_grad_norm=10.0,
    evaluation_strategy="steps",  # Evaluation at regular intervals of steps
    eval_steps=100,  # Evaluate every 100 steps, adjust as needed
    save_strategy="steps",  # Save at regular intervals of steps
    save_steps=100,  # Save every 100 steps, adjust as needed
    save_total_limit=1,  # Keep only the best checkpoint
    load_best_model_at_end=True,
)
# Ensure that each batch in your dataset includes 'input_values'
for batch in train_ds:
    assert INPUT_VALUES in batch, f"'{INPUT_VALUES}' missing from batch"
    assert LABELS in batch, f"'{LABELS}' missing from batch"
    assert batch[INPUT_VALUES] is not None, f"'{INPUT_VALUES}' is None"
    assert batch[LABELS] is not None, f"'{LABELS}' is None"
for batch in train_ds:
    print(type(batch[INPUT_VALUES]))
    assert isinstance(batch[INPUT_VALUES], torch.Tensor), f"'{INPUT_VALUES}' is not a FloatTensor"
    assert batch[INPUT_VALUES].dtype == torch.float, f"'{INPUT_VALUES}' is not a FloatTensor"
    assert isinstance(batch[LABELS], torch.Tensor), f"'{LABELS}' is not a LongTensor"
    assert batch[LABELS].dtype == torch.long, f"'{LABELS}' is not a LongTensor"
print("All batches have the correct keys")

print(train_ds)
train_ds = train_ds.remove_columns(["file", "text", "speaker_id", "chapter_id", "id"])
print(train_ds)
val_ds = val_ds.remove_columns(["file", "text", "speaker_id", "chapter_id", "id"])
test_ds = test_ds.remove_columns(["file", "text", "speaker_id", "chapter_id", "id"])

train_ds = train_ds.rename_column('input_values', 'input_features')
val_ds = val_ds.rename_column('input_values', 'input_features')
test_ds = test_ds.rename_column('input_values', 'input_features')

train_ds = train_ds.rename_column('labels', 'decoder_input_ids')
val_ds = val_ds.rename_column('labels', 'decoder_input_ids')
test_ds = test_ds.rename_column('labels', 'decoder_input_ids')

print(model.forward.__doc__)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

# Automatically find and resume from the latest checkpoint
latest_checkpoint = find_latest_checkpoint(output_dir)
print(f"Resuming from {latest_checkpoint}")
print("Training Model...")
trainer.train(resume_from_checkpoint=latest_checkpoint)
print("Done training Model")

# Save the final model
model.save_pretrained("./final_model")
print("Model saved")

# Evaluate the model
print("Evaluating model...")
results = trainer.evaluate(test_ds)
print("Evaluation results:")
print(results)

import os
import numpy as np
import torch
import shutil
import datetime
import sentencepiece as spm
from evaluate import load
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Speech2TextForConditionalGeneration, Speech2TextConfig, TrainingArguments, Trainer, Speech2TextTokenizer, DataCollatorWithPadding

wer_metric = load("wer")

TRAIN_DATASET = "train.100"

TOKENIZER_PATH = "tokenizer.model"
TOKENIZER_VOCAB_SIZE = 900


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
        # os.remove(TOKENIZER_VOCAB_TRAIN_FILE)
    
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
    batch["input_values"] = embeddings
    batch["labels"] = torch.tensor(labels)

    return batch

def padd_data(percentile_95_audio, percentile_95_labels):
    def _padd_data(batch):
        inputs = batch["input_values"]
        labels = batch["labels"]

        # Pad input_values and labels
        padded_inputs = torch.zeros(percentile_95_audio)
        padded_labels = torch.full((percentile_95_labels,), -100)  # Using -100 which is typically used to ignore in loss computation

        # Copy the data to the padded tensors
        seq_len = min(percentile_95_audio, inputs.shape[0])
        label_len = min(percentile_95_labels, labels.shape[0])

        padded_inputs[:seq_len] = inputs[:seq_len]
        padded_labels[:label_len] = labels[:label_len]

        # Update the batch
        batch["input_values"] = padded_inputs
        batch["labels"] = padded_labels

        return batch
    return _padd_data

def calculate_percentiles(processed_dataset):

    # Calculate lengths
    audio_lengths = [len(data['input_values']) for data in processed_dataset]
    label_lengths = [len(data['lables']) for data in processed_dataset]

    # Calculate 95th percentiles
    percentile_95_audio = np.percentile(audio_lengths, 95)
    percentile_95_labels = np.percentile(label_lengths, 95)

    return percentile_95_audio, percentile_95_labels


def load_and_preprocess_data(split, dataset_path):
    dataset_file = f"{dataset_path}/{split}_processed.pt"
    if os.path.isfile(dataset_file):
        print(f"Loading processed {split} dataset from {dataset_file}")
        return torch.load(dataset_file)
    else:
        print(f"Loading and processing {split} dataset...")
        dataset = load_dataset("librispeech_asr", "clean", split=split)

        # TODO Optional: Subset the dataset if needed
        dataset = dataset.select(range(50))  # Select first 50 samples for testing

        processed_dataset = dataset.map(prepare_data, remove_columns=["audio"])

        print("Calculating 95th percentiles...")        
        percentile_95_audio, percentile_95_labels = calculate_percentiles(processed_dataset)
        print(f"95th percentile audio length: {percentile_95_audio}")
        print(f"95th percentile label length: {percentile_95_labels}")
        print("Padding data...")        
        padded_dataset = processed_dataset.map(padd_data(percentile_95_audio, percentile_95_labels))
       
        print(f"Processed {split} dataset, saving to {dataset_file}...")
        torch.save(padded_dataset, dataset_file)
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


class CustomDataCollator:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        print(batch[0].keys())
        # Find the longest sequence in the batch
        max_length = self.max_length or max(len(x['input_values']) for x in batch)

        # Pad all sequences to the length of the longest sequence
        for item in batch:
            input_length = len(item['input_values'])
            item['input_values'] = item['input_values'] + [0] * (max_length - input_length)  # Adjust padding value if needed
            item['labels'] = item['labels'] + [self.tokenizer.pad_id()] * (max_length - len(item['labels']))

        # Create tensors
        input_values = torch.tensor([x['input_values'] for x in batch], dtype=torch.float32)
        labels = torch.tensor([x['labels'] for x in batch], dtype=torch.long)

        return {"input_values": input_values, "labels": labels}


data_collator = CustomDataCollator(tokenizer=tokenizer)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
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

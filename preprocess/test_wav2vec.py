from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2FeatureExtractor
import torch
from jiwer import wer


librispeech_eval = load_dataset("librispeech_asr", "clean", split="test[10:20]")

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self") #.to("cuda")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
#model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h") #.to("cuda")
#processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

def map_to_pred(batch):
    inputs = processor(batch["audio"]["array"], return_tensors="pt", padding="longest")
    input_values = inputs.input_values#.to("cuda")
    attention_mask = inputs.attention_mask#.to("cuda")
    
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
        #logits = model(input_values).logits
        #last_hidden = model(input_values, output_hidden_states=True).last_hidden_state

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription[0]
    #batch["last_hidden"] = last_hidden
    return batch


result = librispeech_eval.map(map_to_pred, remove_columns=["audio"])

print(list(zip(result["text"], result["transcription"])))

print("WER:", wer(result["text"], result["transcription"]))

#print(result["last_hidden"])
#print(result["last_hidden"][0])

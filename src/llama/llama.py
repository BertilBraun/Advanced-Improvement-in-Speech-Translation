import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig

from src.logger_utils import get_logger

logger = get_logger('LLaMA::llama')


logger.info('Loading LLaMA...')

LLAMA_MODEL = 'meta-llama/Llama-2-7b-chat-hf'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logger.info(f'Using device: {DEVICE}')
logger.info('Loading Tokenizer')

TOKENIZER = AutoTokenizer.from_pretrained(LLAMA_MODEL, padding_side='left')
TOKENIZER.pad_token = TOKENIZER.eos_token

logger.info('Tokenizer ready.')
logger.info('Loading LLaMA model.')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16
)

LLM = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL, quantization_config=bnb_config, device_map='auto')

logger.info('Configuring LLaMA model.')

LLM.config.pad_token_id = TOKENIZER.pad_token_id = 0  # unk
LLM.config.bos_token_id = 1  # bos
LLM.config.eos_token_id = 2  # eos

# bnb config apparently already moves to the correct device LLM = LLM.to(DEVICE)

logger.info('LLaMA ready.')


def generate(prompts: list[str]) -> list[str]:
    """Generates text for each prompt in the given list of prompts. Returns a list of generated texts in the same order as the prompts."""
    model_inputs = TOKENIZER(prompts, return_tensors='pt', padding=True).to(DEVICE)

    generation_config = GenerationConfig(
        temperature=0.5,  # TODO really low temperature but still get hallucinations
        top_p=0.75,
        repetition_penalty=1.1,
        do_sample=True,
        num_beams=1,
    )

    max_new_tokens = model_inputs.input_ids.shape[-1] * 2

    with torch.inference_mode():
        generated_ids = LLM.generate(
            **model_inputs,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
        )

    decoded_outputs = TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)
    decoded_inputs = TOKENIZER.batch_decode(model_inputs.input_ids, skip_special_tokens=True)

    cleaned_outputs = [output.replace(prompt, '') for output, prompt in zip(decoded_outputs, decoded_inputs)]

    with open('llama_outputs.txt', 'a', encoding='utf-8') as f:
        for output, prompt in zip(cleaned_outputs, decoded_inputs):
            f.write(f'{prompt}\n{output}\n\n')
            f.write('-' * 100 + '\n\n')

    return cleaned_outputs

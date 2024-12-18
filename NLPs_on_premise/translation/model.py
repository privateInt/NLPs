from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
# from transformers import TextStreamer
import torch
import json
import asyncio

class translator:
    def __init__(self, model_path: str, max_seq_length: int, dtype=None, load_in_4bit=True):
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit

        # Load model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)

        # Set up tokenizer
        self.tokenizer = get_chat_template(self.tokenizer, chat_template="llama-3.1")
        # self.text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)

    async def inference(self, system_message: str, user_input: str):
        prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input},
        ]

        inputs = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")

        outputs = await asyncio.to_thread(
            self.model.generate,
            inputs['input_ids'],
            max_new_tokens=64,
            use_cache=True,
            temperature=0.5,
            repetition_penalty=2.0
        )

        result = self.tokenizer.batch_decode(outputs)
        final = result[0].split("<|end_header_id|>")[-1].split("<|reserved_special_token_")[0]
        
        return str(final.strip())
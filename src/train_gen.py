import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import torch
from prompt_template import *
from args import get_args_gen
from utils import set_seeds
import sys


def load_dataset(json_file, tokenizer, mode='classify'):
    inst_list = []
    data = []
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            inst_list.append(json.loads(line))

    for item in inst_list:
        assistant_output = json.dumps(item, ensure_ascii=False, indent=2)

        if mode == 'classify':
            messages = [
                {"role": "system", "content": system_gen},
                {"role": "user", "content": user_gen.format(text=item["text"])},
                {"role": "assistant", "content": item["action"]}
            ]
        elif mode == 'annotation':
            messages = [
                {"role": "system", "content": system_annotation},
                {"role": "user", "content": user_annotation.format(text=item['annotation']["text"])},
                {"role": "assistant", "content": assistant_output}
            ]

        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        data.append({"text": full_text})

    return Dataset.from_list(data)


def setup_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj","v_proj"]
    )

    model = get_peft_model(model, peft_config)

    return model, tokenizer

def main():
    args = get_args_gen()
    set_seeds(args.seed)
    model, tokenizer = setup_model_and_tokenizer(args)

    dataset = load_dataset(args.data_path, tokenizer, mode=args.mode)

    training_args = TrainingArguments(
        output_dir="./tmp",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.n_epochs,
        logging_dir="./log",
        logging_steps=50,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=model.peft_config['default'],
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()

    trainer.save_model(args.output_path)
    tokenizer.save_pretrained(args.output_path)


if __name__ == "__main__":
    main()
from google import genai
from google.genai import types
from prompt_template import *
import json
from utils import clean_text
import random
from datasets import Dataset, DatasetDict


client = genai.Client(api_key="")

label2id = {
    "translate_up": 0,
    "translate_down": 1,
    "translate_left": 2,
    "translate_right": 3,
    "rotate_up": 4, 
    "rotate_down": 5,
    "rotate_left": 6,
    "rotate_right": 7,
    "move_forward": 8,
    "move_backward": 9,
    "out_of_scope": 10
}
id2label = {
    0: "translate_up",
    1: "translate_down",
    2: "translate_left",
    3: "translate_right",
    4: "rotate_up", 
    5: "rotate_down",
    6: "rotate_left",
    7: "rotate_right",
    8: "move_forward",
    9: "move_backward",
    10: "out_of_scope"
}

def generate_instructions(system_instruction, prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
            temperature=0.8,
            top_p=0.9,
        ),
    )
    instructions = clean_text(response.text)
    return instructions


def create_dataset(args, val_ratio=0.1):
    # 读取原始 JSON
    with open("./data/aug_instructions.jsonl", 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 转换为扁平结构
    examples = []
    for entry in raw_data:
        action = entry["action"]
        label = label2id[action]
        for instr in entry["instruction"]:
            text = instr["text"].strip()
            examples.append({"text": text, "label": label})

    # 打乱数据并划分
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - val_ratio))
    train_data = examples[:split_idx]
    val_data = examples[split_idx:]

    # 转为 HuggingFace Datasets 对象
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data)
    })

    return dataset


def data_augumentation():
    data = []

    with open("./data/instructions.json", 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    for entry in raw_data:
        action = entry["action"]
        label = label2id[action]
        for instr in entry["instruction"]:
            text = instr["text"].strip()
            
            user_instruction = user_instruction_aug.format(
                original_command=text, 
            )
            aug_instruction = generate_instructions(system_instruction_aug, user_instruction)
            for aug_instr in aug_instruction:
                data.append({"text": aug_instr['augmented_text'], "label": label})
    
    with open("./data/aug_instructions.jsonl", "w", encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def generate_raw():
    responses = []
    for action, examples in example_instructions.items():
        instruction = user_instruction.format(
            action=action, 
            example_1=examples["example_1"],
            example_2=examples["example_2"],
            example_3=examples["example_3"],
            example_4=examples["example_4"],
        )
        response = generate_instructions(system_instruction, instruction)
        responses.append({"action": action, "instruction": response})
    
    response = generate_instructions(system_instruction_oos, user_instruction_oos)
    responses.append({"action": "out_of_scope", "instruction": response})

    with open("./data/instructions.json", "w", encoding='utf-8') as f:
        json.dump(responses, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # generate_raw()
    data_augumentation()
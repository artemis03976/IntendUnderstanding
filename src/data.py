from google import genai
from google.genai import types
from prompt_template import *
import json
from utils import clean_text, split_list
import random
from datasets import Dataset, DatasetDict
import sys


# client = genai.Client(api_key="AIzaSyD61W-Ih2h_BFeHidOtGOuvBx5mXaH8K7U")

client = genai.Client(
    api_key="sk-zk28949bec452c89fa684cc2a50acff47cc687dcd8e7cb8a",
    http_options={
        "base_url": "https://api.zhizengzeng.com/google"
    },
)

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
        model="gemini-2.5-pro", 
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            # thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
            temperature=0.9,
            top_p=0.9,
        ),
    )
    instructions = clean_text(response.text)
    if instructions is None:
        print(response.text)
    return instructions


def create_dataset(args, val_ratio=0.1):
    # 读取原始 JSON
    examples = []
    with open("./data/instructions.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            examples.append({"text": item['text'], "label": item['action']})

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
    
    response = generate_instructions(system_oos, user_oos)
    responses.append({"action": "out_of_scope", "instruction": response})

    with open("./data/instructions.json", "w", encoding='utf-8') as f:
        json.dump(responses, f, indent=4, ensure_ascii=False)


def generate_aug(system_template, user_template, file_name):
    new_data = []
    for action, examples in example_instructions.items():
        instruction_aug = user_template.format(
            action=action, 
        )
        response = generate_instructions(system_template, instruction_aug)
        new_data.append({"action": action, "instruction": response})

        with open(os.path.join("./data/", file_name), "w", encoding='utf-8') as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)


def generate_structured():
    raw_data = []
    new_data = []

    with open("./data/instructions.jsonl", 'r', encoding='utf-8') as f:
        for item in f:
            raw_data.append(json.loads(item))
    
    inst_list = raw_data.copy()
    random.shuffle(inst_list)

    # Temporary save
    with open(os.path.join("./data/", "tmp.jsonl"), "w", encoding='utf-8') as f:
        for item in inst_list:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    inst_list = split_list(inst_list, 20)

    for inst in inst_list:
        instruction_structured = user_structured.format(
            inst_list=inst, 
        )
        response = generate_instructions(system_structured, instruction_structured)
        new_data.extend(response)

        with open(os.path.join("./data/", "structured.jsonl"), "w", encoding='utf-8') as f:
            for item in new_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Save {len(new_data)} instructions to structured.jsonl")


def generate_speech():
    data = []

    with open("./data/adv_instructions.json", 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    for entry in raw_data:
        action = entry["action"]
        label = label2id[action]
        for instr in entry["instruction"]:
            text = instr["text"].strip()
            
            user_instruction = user_speech.format(
                original_command=text, 
            )
            aug_instruction = generate_instructions(system_speech, user_instruction)
            for aug_instr in aug_instruction:
                data.append({"text": aug_instr['augmented_text'], "label": label})
    
    with open("./data/aug_instructions.jsonl", "w", encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def convert_to_jsonl():
    with open("./data/instructions.json", 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    inst_list = []
    for item in raw_data:
        action = item["action"]
        instructions = item["instruction"]

        for sample in instructions:
            inst_list.append({'action': action, 'text': sample["text"]})
    
    with open(os.path.join("./data/", "instructions.jsonl"), "w", encoding='utf-8') as f:
        for item in inst_list:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


if __name__ == "__main__":
    # generate_raw()
    
    # generate_aug(system_inv, user_inv, "inv.json")
    # generate_aug(system_adv, user_adv, "adv.json")
    # generate_aug(system_compound, user_compound, "compound.json")
    # generate_aug(system_imp, user_imp, "imp.json")
    generate_structured()

    # generate_speech()

    # convert_to_jsonl()

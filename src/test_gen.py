import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
import warnings
from prompt_template import *
from args import get_test_args
from utils import set_seeds

warnings.filterwarnings("ignore")


class GenerativeIntentClassifier:
    def __init__(self, base_model_id, adapter_path, mode='classify'):
        print("--- 正在加载基础模型和分词器... ---")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- 使用设备: {self.device} ---")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"--- 正在从 {adapter_path} 加载LoRA适配器... ---")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)

        # 设置为评估模式
        self.model.eval()
        print("--- 模型加载并准备完毕！ ---")

        self.mode = mode

    def format_prompt(self, user_text):
        if self.mode == 'classify':
            messages = [
                {"role": "system", "content": system_gen},
                {"role": "user", "content": user_gen.format(text=user_text)},
            ]
        elif self.mode == 'annotation':
            messages = [
                {"role": "system", "content": system_annotation},
                {"role": "user", "content": user_annotation.format(text=user_text)},
            ]

        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt

    def predict(self, user_text):
        prompt = self.format_prompt(user_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        assistant_marker = "assistant\n"
        if assistant_marker in full_response:
            prediction = full_response.split(assistant_marker)[-1].strip()
    
        return prediction


def main():
    args = get_test_args()
    set_seeds(args.seed)
    classifier = GenerativeIntentClassifier(args.model_name, args.checkpoint_path, mode=args.mode)
    
    print("\n" + "="*50)
    print(" 语音指令意图识别测试脚本 ")
    print(" 输入 'exit' 或 '退出' 来结束程序")
    print("="*50 + "\n")
    
    while True:
        input_text = input("请输入指令: ")
        
        if input_text.lower() in ["exit", "退出"]:
            print("--- 程序已退出 ---")
            break
            
        if not input_text.strip():
            continue
            
        # 调用预测函数
        predicted_label = classifier.predict(input_text)
        
        # 打印结果
        print(f"  > 模型输出: {predicted_label}")
        print("-" * 30)


if __name__ == "__main__":
    main()
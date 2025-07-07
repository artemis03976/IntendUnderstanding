import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F
from data import id2label
from args import get_test_args
from utils import set_seeds


class IntentClassifier:
    def __init__(self, model_path: str):
        """
        初始化模型、分词器和设备
        :param model_path: 保存的模型文件夹路径
        :param id2label: ID到标签的映射字典
        """
        print("--- 正在加载模型和分词器... ---")
        # 自动选择设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- 使用设备: {self.device} ---")

        # 加载分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = id2label
        print("--- 模型加载完毕，准备进行推理 ---")

    def predict(self, text: str):
        """
        对输入的单句文本进行意图预测
        :param text: 需要预测的文本字符串
        :return: (预测的标签, 置信度分数)
        """

        with torch.no_grad():
            # 1. 使用分词器对输入文本进行编码
            # return_tensors="pt" 表示返回PyTorch tensors
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            # 2. 将编码后的数据移动到与模型相同的设备
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # 3. 模型前向传播，得到logits
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 4. 使用Softmax函数将logits转换为概率分布
            probabilities = F.softmax(logits, dim=-1)
            
            # 5. 获取最高概率的预测结果
            # torch.max会返回最大值和对应的索引
            confidence, predicted_id_tensor = torch.max(probabilities, dim=-1)
            
            # 从tensor中提取数值
            predicted_id = predicted_id_tensor.item()
            confidence_score = confidence.item()
            
            # 6. 使用映射字典找到对应的文本标签
            predicted_label = self.id2label.get(predicted_id, "未知标签")
            
            return predicted_label, confidence_score


def main():
    args = get_test_args()
    set_seeds(args.seed)
    classifier = IntentClassifier(args.checkpoint_path)

    print("\n" + "="*50)
    print(" 语音指令意图识别测试脚本 ")
    print(" 输入 'exit' 或 '退出' 来结束程序")
    print("="*50 + "\n")
    
    # 进入一个无限循环，接收用户输入
    while True:
        # 提示用户输入
        input_text = input("请输入要测试的指令: ")
        
        # 检查退出条件
        if input_text.lower() in ["exit", "退出"]:
            print("--- 程序已退出 ---")
            break
            
        # 如果输入为空，则继续下一次循环
        if not input_text.strip():
            continue
            
        # 调用预测函数
        label, score = classifier.predict(input_text)
        
        # 打印结果
        print(f"  > 预测结果: {label} (置信度: {score:.2%})")
        print("-" * 30)


if __name__ == "__main__":
    main()

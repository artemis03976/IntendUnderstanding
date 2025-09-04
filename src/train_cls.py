from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from args import get_args_cls
from utils import set_seeds
from data import create_dataset


def compute_metrics(eval_pred):
    """计算评估指标"""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    # 计算精确率、召回率、F1分数（宏平均）
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    # 计算准确率
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    args = get_args_cls()
    set_seeds(args.seed)

    model = DistilBertForSequenceClassification.from_pretrained(args.model_name, num_labels=args.n_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    dataset = create_dataset(args)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_path,                   # 训练结果输出目录
        num_train_epochs=args.n_epochs,                # 训练轮次
        per_device_train_batch_size=args.batch_size,   # 训练时的batch size
        per_device_eval_batch_size=args.batch_size,    # 评估时的batch size
        warmup_steps=50,                              # 预热步数
        weight_decay=0.01,                             # 权重衰减
        logging_dir=args.log_path,                     # 日志目录
        logging_steps=30,
        evaluation_strategy="epoch",                   # 每个epoch结束后进行一次评估
        save_strategy="epoch",                         # 每个epoch结束后保存一次模型
        load_best_model_at_end=True,                   # 训练结束后加载最佳模型
        metric_for_best_model="f1",                    # 以f1分数作为评判最佳模型的标准
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(eval_results)

    trainer.save_model(args.output_path)


if __name__ == "__main__":
    main()
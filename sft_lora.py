import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import os

def load_training_data(file_path):
    """加载训练数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def format_prompt(instruction, input_text, output):
    """格式化提示词"""
    if input_text:
        prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n{output}"
    else:
        prompt = f"### 指令:\n{instruction}\n\n### 回答:\n{output}"
    return prompt

def prepare_dataset(data, tokenizer, max_length=512):
    """准备数据集"""
    formatted_data = []
    
    for item in data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        
        # 格式化提示词
        prompt = format_prompt(instruction, input_text, output)
        
        # 编码
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        
        # 设置labels，但忽略填充token
        labels = encoded['input_ids'].copy()
        for i, token_id in enumerate(encoded['input_ids']):
            if token_id == tokenizer.pad_token_id:
                labels[i] = -100  # 忽略填充token的损失
        encoded['labels'] = labels
        
        formatted_data.append(encoded)
    
    return Dataset.from_list(formatted_data)

def main():
    # 模型路径
    model_path = "../model"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型路径 {model_path} 不存在")
        return
    
    print("开始加载模型和分词器...")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("模型加载完成！")
    
    # 配置LoRA
    print("配置LoRA...")
    lora_config = LoraConfig(
        r=64,  # LoRA秩
        lora_alpha=16,  # LoRA缩放参数
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,  # LoRA dropout
        bias="none",  # 不训练bias
        task_type=TaskType.CAUSAL_LM  # 因果语言模型任务
    )
    
    # 应用LoRA配置
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数数量
    model.print_trainable_parameters()
    
    # 加载训练数据
    print("加载训练数据...")
    training_data = load_training_data("training_data.json")
    print(f"加载了 {len(training_data)} 条训练数据")
    
    # 准备数据集
    print("准备数据集...")
    dataset = prepare_dataset(training_data, tokenizer)
    
    # 分割训练集和验证集
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./lora_output",
        num_train_epochs=5,  # LoRA训练可以更多epoch
        per_device_train_batch_size=2,  # LoRA显存占用小，可以增大批次
        gradient_accumulation_steps=4,
        learning_rate=1e-4,  # LoRA通常使用更大的学习率
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,  # 禁用wandb等报告工具
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("开始LoRA训练...")
    trainer.train()
    
    # 保存LoRA模型
    print("保存LoRA模型...")
    trainer.save_model("./lora_final_model")
    tokenizer.save_pretrained("./lora_final_model")
    
    # 保存LoRA适配器
    model.save_pretrained("./lora_adapter")
    
    print("LoRA训练完成！")
    print("模型已保存到 ./lora_final_model")
    print("LoRA适配器已保存到 ./lora_adapter")

if __name__ == "__main__":
    main() 
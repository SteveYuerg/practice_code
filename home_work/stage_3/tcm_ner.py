from functools import partial
import json
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from swanlab.integration.transformers import SwanLabCallback
import swanlab
from modelscope import AutoTokenizer

from peft import LoraConfig, get_peft_model, TaskType

import torch
import random
import numpy as np
import os

seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)



model_id = "qwen2.5-1.5B-Instruct"  
model_path = "G:/model_weights/models/model/qwen2.5-1.5B-Instruct"
data_path = "D:/doc/my_project/lianxi/home_work/stage_3/data_set"

class TCMNERDataset():
    def __init__(self, data_path):
        self.label_set = set()
        self.data = self.load_data(data_path)
        # self.dataset = Dataset.from_list(self.data)
        self.id2label = {idx: item for idx, item in enumerate(self.label_set)}
        self.label2id = {item: idx for idx, item in self.id2label.items()}


    def load_data(self, data_path):
        dataset = []
        with open(data_path, 'r', encoding='utf-8') as f:
            trunk = f.read().split('\n\n')
            for line in trunk:
                input, output = '', []
                for idx, string in enumerate(line.split('\n')):
                    if not string.strip():  # 跳过空行
                        continue
                    text, tag = string.split(' ')  # 假设文本和标签之间用空格分隔
                    input += text
                    if tag.startswith('B-'):
                        self.label_set.add(tag)
                        output.append([idx, 0, text, tag[2:]])  # 添加标签内容
                    elif tag.startswith('I-') and output:
                        self.label_set.add(tag)
                        output[-1][1] = idx  # 更新标签的结束位置
                        output[-1][2] += text  # 更新标签内容
                dataset.append({'input': input, 'output': json.dumps(output, ensure_ascii=False)})  # 将标签列表转换为JSON字符串
        return dataset
    

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)#, id2label=train_data.id2label, label2id=train_data.label2id)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    
from pathlib import Path
train_data_path = Path(data_path) / 'medical.train'
test_data_path = Path(data_path) / 'medical.test'
dev_data_path = Path(data_path) / 'medical.dev'

train_data = TCMNERDataset(train_data_path)
test_data = TCMNERDataset(test_data_path)
dev_data = TCMNERDataset(dev_data_path)

train_dataset = Dataset.from_list(train_data.data)
test_dataset = Dataset.from_list(test_data.data)
dev_dataset = Dataset.from_list(dev_data.data)

def preprocess_function(tokenizer, examples):
    MAX_LENGTH = 512
    system_prompt = "你是一个中医药领域的命名体识别专家，你需要从给定句子中提取相关命名体，以json格式输出，例如：[[', 4, '口苦', '临床表现'],[5, 6, '口干', '临床表现']]，其中每个元素分别表示命名体的起始位置、结束位置、文本内容和标签类型。"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": examples['input']},
    ]
    messages = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,) # 输出会停在最后一个user，然后添加assistant的开始标记等待模型生成, 不会编码assistant的内容
    # print(messages)
    # print(examples)
    response = tokenizer(examples['output'], add_special_tokens=False)
    inputs = tokenizer(messages, truncation=True, add_special_tokens=False)

    input  = inputs['input_ids'] + response['input_ids'] + [tokenizer.pad_token_id]  # 将输入和输出拼接在一起，并添加结束标记

    attention_mask = inputs['attention_mask'] + response['attention_mask'] + [1]  # 拼接注意力掩码

    outputs = [-100] * len(inputs['input_ids']) + response['input_ids'] + [tokenizer.eos_token_id]  # 输出标签的token id，并添加结束标记

    if len(input) > MAX_LENGTH:
        input = input[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        outputs = outputs[:MAX_LENGTH]

    return {'input_ids': input, 'attention_mask': attention_mask, 'labels': outputs}

p_preprocess = partial(preprocess_function, tokenizer)

train_datasets = train_dataset.map(p_preprocess, remove_columns=train_dataset.column_names, num_proc=4)
# test_datasets = test_dataset.map(p_preprocess, remove_columns=test_dataset.column_names, num_proc=4)
dev_datasets = dev_dataset.map(p_preprocess, remove_columns=dev_dataset.column_names, num_proc=4)

fine_tuning_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                r=8, 
                                lora_alpha=16, 
                                inference_mode=False, 
                                lora_dropout=0.1)

peft_model = get_peft_model(model, fine_tuning_config)
peft_model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="G:/model_weights/models/checkpoint/qwen2.5-1.5-tcm-ner",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4, # 积累多少batch后更新一次权重，主要用于解决显存不足的问题
    eval_strategy="epoch",
    logging_steps=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)


swanlab_callback = SwanLabCallback(
    project="Qwen2.5-NER-fintune",
    experiment_name="Qwen2-1.5B-Instruct",
    description="使用通义千问Qwen2-1.5B-Instruct模型在中医药NER数据集上微调，实现关键实体识别任务。",
    config={
        "model": model_id,
        "model_dir": model_path,
        "dataset": "qgyd2021/chinese_ner_sft",
    },
)

trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_datasets,
    eval_dataset=dev_datasets,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)


trainer.train()

swanlab.finish()
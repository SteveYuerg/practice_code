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
checkpoint_path = "G:/model_weights/models/checkpoint/qwen2.5-1.5-tcm-ner/checkpoint-658"

# model_path = "/root/autodl-tmp/qwen2.5-7B-Instruct"
# data_path = "./data_set"
# checkpoint_path = "/root/autodl-tmp/checkpoint/qwen2.5-7B-tcm-ner3/checkpoint-658"

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
                        self.label_set.add(tag[2:])
                        output.append([text, tag[2:]])  # 添加标签内容
                    elif tag.startswith('I-') and output:
                        output[-1][0] += text  # 更新标签内容
                dataset.append({'input': input, 'output': json.dumps(output, ensure_ascii=False)})  # 将标签列表转换为JSON字符串
        return dataset
    
from pathlib import Path
test_data_path = Path(data_path) / 'medical.test'

test_data = TCMNERDataset(test_data_path)

from peft import  PeftModel
# base_model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)
# peft_model = PeftModel.from_pretrained(base_model, checkpoint_path).cuda()

peft_model = AutoModelForCausalLM.from_pretrained(model_path).cuda()

from pathlib import Path
test_data_path = Path(data_path) / 'medical.test'

test_data = TCMNERDataset(test_data_path)
test_dataset = Dataset.from_list(test_data.data)


def f1_score(set_a, set_b):
    """ 严格的F1 分数计算，只有当实体、标签预测完全正确时才算TP，否则算FP和FN """
    TP = len(set_a & set_b)
    FP = len(set_b - set_a)
    FN = len(set_a - set_b)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

import tqdm

set_a_strict = set()
set_b_strict = set()
set_a_relax = set()
set_b_relax = set()


tag_dict = {}
for entity in test_data.label_set:
    tag_dict[entity] = [set(), set()]  # [label, label_predicted]

def predict_collect(predictions, references):
    """ 预测结果收集函数，返回严格匹配和宽松匹配的实体集合 """
    set_a_strict = set()  
    set_b_strict = set()  
    
    for item in references:
        item = json.loads(item)
        for label in item:
            tag_dict[label[1]][0].add(tuple(label))  # 仅比较实体内容和标签类型，忽略位置
            set_a_strict.add(tuple(label))

    for output in predictions:
        try:
            output = output.strip().replace('\n', '').replace('“', '"').replace('”', '"').replace('{', '').replace('}', '')
            output = json.loads(output)  # 去除多余的换行符和空格

            for item in output:
                if not item or len(item) != 2:  # 如果输出是空列表，跳过
                    continue
                if item[1] in tag_dict:
                    tag_dict[item[1]][1].add(tuple(item))
                set_b_strict.add(tuple(item))
        except:
            import traceback
            # print('--1')
            # print(output)
            # print('--2')
            # traceback.print_exc()
    
    return set_a_strict, set_b_strict,

import tqdm

from torch.utils.data import DataLoader

test_dataset = DataLoader(test_data.data, batch_size=32, shuffle=False)

system_prompt = '你是一个中医药领域的命名体识别专家，你需要从给定句子中提取相关命名体，以json格式输出，例如：[["口苦", "临床表现"],["口干", "临床表现"]]，其中每个元素分别表示实体和标签类型。如果检测不到实体则返回空列表[]'

pbar = tqdm.tqdm(total=len(test_dataset), desc="Evaluating on test set")

for batch_sample in test_dataset:
    inputs = []
    for i in batch_sample['input']:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": i},
        ]
        input = tokenizer.apply_chat_template(messages, padding_side='left', tokenize=False, add_generation_prompt=True) # 输出会停在最后一个user，然后添加assistant的开始标记等待模型生成, 不会编码assistant的内容
        inputs.append(input)
    inputs = tokenizer(inputs, padding_side='left', padding=True, truncation=True, return_tensors="pt").to(peft_model.device)

    with torch.no_grad():
        outputs = peft_model.generate(inputs.input_ids, max_new_tokens=512)
    
    output = outputs[:, inputs.input_ids.shape[1]:]  # 获取生成的部分
    decoded_outputs = tokenizer.batch_decode(output, skip_special_tokens=True)

    set_a_strict, set_b_strict = predict_collect(decoded_outputs, batch_sample['output'])

print(f"Strict F1 Score: {f1_score(set_a_strict, set_b_strict):.4f}")

for entity, (set_a, set_b) in tag_dict.items():
    entity_f1 = f1_score(set_a, set_b)
    print(f"Entity: {entity}, F1 Score: {entity_f1:.4f}")
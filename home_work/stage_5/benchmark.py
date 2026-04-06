import time
import random 
import uuid
from locust import HttpUser, task, between
from datasets import load_dataset

data_path = 'G:/data_set/minimind/sft_mini_512.jsonl'
data_path = 'G:/data_set/minimind/rlaif-mini.jsonl'
dataset = load_dataset('json', data_files=data_path, split='train')

test_samples = [i['conversations'][0]['content'] for i in dataset]
print(test_samples[:4])

stats = {
    "total_requests": 0,
    "total_tokens": 0,
    "start_time": time.time(),
    "end_time": time.time()
}

class User(HttpUser):
    wait_time = between(1, 1.5)

    @task
    def task_post_archive(self):
        trace_id = f'cevi{uuid.uuid4().hex}'
        test_server = 'http://localhost:11434'
        path = '/api/chat'
        url = f'{test_server}{path}'
        headers = {
            'Content-Type': 'application/json',
        }
        data = {
            "model": "qwen3:4b", 
            "query": random.choice(test_samples),
            "trace_id": trace_id,
        }
        self.client.post(url, json=data, headers=headers)


if __name__ == "__main__":
    pass
    import requests
    url = "http://localhost:11434/api/chat"
    data = {
        "model": "qwen3:4b", 
        "messages": [
            {"role": "user", "content": "你好，你能帮我写一段 Python 代码吗？"}
        ],
        "stream": True
    }
    response = requests.post(url, json=data)
    print("对话回复:", response.json())
    res = response.json()
    res.pop("message")
    print(res)
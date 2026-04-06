from torch.utils.data import DataLoader
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

MODEL_ID = "G:/model_weights/models/Qwen/qwen2.5-1.5B-Instruct"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

kv_cache_scheme = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )
recipe = QuantizationModifier(
    targets="Linear",
    # scheme="W4A16",
    kv_cache_scheme=kv_cache_scheme,
    ignore=["lm_head", "re:.*mlp.gate$"],
)

data_path = "G:/data_set/minimind/rlaif-mini.jsonl"
# from torch.utils.data import Dataset
from datasets import load_dataset, Dataset

dataset = load_dataset("json", data_files=data_path, split="train")

data_col = []
for data in dataset:
    conversations = data["conversations"]
    text = tokenizer.apply_chat_template(conversations, max_length=512, tokenize=False, add_generation_prompt=False)
    data_col.append(tokenizer(text, padding=True, truncation=True, max_length=512))

dataset = Dataset.from_list(data_col)
print(dataset[0])

# Apply quantization.
oneshot(model=model, recipe=recipe, dataset=dataset)#, num_calibration_samples=10000)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = "G:/model_weights/models/Qwen/qwen2.5-1.5B-Instruct-KVCache-FP8"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
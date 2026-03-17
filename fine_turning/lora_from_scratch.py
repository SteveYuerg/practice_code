from copy import deepcopy

import torch 
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel

model_path = "G:/model_weights/huggingface_model/Qwen3-4B"
# model_path = "G:/model_weights/huggingface_model/models--Langboat--bloom-1b4-zh/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)
config = AutoConfig.for_model('llama')
config.hidden_size = 24
config.intermediate_size = config.hidden_size * 4
config.num_attention_heads = 4
config.num_hidden_layers = 4
config.num_key_value_heads = 2
config.vocab_size = 128

model = AutoModel.from_config(config)  # 没带因果头

class LoraLinear(nn.Module):
    def __init__(self, 
                 base_layer: nn.Linear,
                 r: int = 8,
                 alpha: int = 16,
                 dropout_p: float = 0,
                 test_mode = False
                 ) -> None:
        super(LoraLinear, self).__init__()
        self.base_layer = deepcopy(base_layer)
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_p)

        self.Lora_A = nn.Parameter(torch.empty((r, base_layer.in_features), dtype=base_layer.weight.dtype))
        self.Lora_B = nn.Parameter(torch.empty((base_layer.out_features, r), dtype=base_layer.weight.dtype))

        nn.init.normal_(self.Lora_A, mean=0.0, std=0.02)

        if test_mode:
            nn.init.normal_(self.Lora_A, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(self.Lora_B)

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        scaling = float(self.alpha) / float(self.r)
        lora_adjustment = F.linear(self.dropout(x), self.Lora_A)
        lora_adjustment = F.linear(lora_adjustment, self.Lora_B)
        return self.base_layer(x) + scaling * lora_adjustment

        
def linear_model_detect(model_name):
    for s in ['embed', 'norm', 'lm_head']:
        if s in model_name:
            return True, s
    return False, None
    
def replace_linear_with_lora(
    module: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout_p = 0.0,
    embed_requires_grad: bool = False,
    normal_requires_grad: bool = False,
    head_requires_grad:bool = False,
    test_mode: bool = False
):
    def linear_model_detect(model_name):
        grad_map = {
            'norm': normal_requires_grad,
            'embed': embed_requires_grad,
            'lm_head': head_requires_grad,
        }
        for s in grad_map.keys():
            if s in model_name:
                return True, grad_map[s]
        return False, False

    for name, child in module.named_children():
        exist, grad_set = linear_model_detect(name)
        if exist: 
            for param in child.parameters():
                param.requires_grad = grad_set
        elif isinstance(child, nn.Linear):
            lora_linear = LoraLinear(child, r=r, alpha=alpha, dropout_p=dropout_p, test_mode=test_mode)
            setattr(module, name, lora_linear)
        else:
            replace_linear_with_lora(
                child, r=r, alpha=alpha, dropout_p=dropout_p,
                embed_requires_grad=embed_requires_grad,
                normal_requires_grad=embed_requires_grad,
                head_requires_grad=head_requires_grad
            )
            
            
            
def print_trainable_parameters(module: nn.Module):
    total_params = sum(p.numel() for p in module.parameters())
    trainable_parameters = sum(p.numel() for p in module.parameters() if p.requires_grad)
    trainable_percentage = trainable_parameters / total_params
    print(f"trainable params: {trainable_parameters:,} || all params: {total_params:,} || trainable%: {trainable_percentage:.4f}")
    
print_trainable_parameters(model)
lora_model = deepcopy(model).to('cuda')
replace_linear_with_lora(lora_model, r=8, alpha=16)
print_trainable_parameters(lora_model)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r = 8,
    lora_alpha = 16,
    target_modules='all-linear'
)
peft_lora_model = deepcopy(model).to('cuda')
peft_lora_model = get_peft_model(peft_lora_model, lora_config)
peft_lora_model.print_trainable_parameters()
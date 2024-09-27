import torch
import torch.nn as nn

from models.llm2vec import LLM2Vec
from utils import colorstr


class AutomaticCodingModel(nn.Module):
    def __init__(self, config, device):
        super(AutomaticCodingModel, self).__init__()
        self.pooling_mode = config.pooling_mode
        self.max_length = config.max_len
        self.l2v, self.tokenizer = self._init_l2v(config, device)
        self.fc_layer = nn.Linear(4096, config.n_labels, bias=False)

        # Change parameter's requires_grad
        if config.lora_activate:
            self._chage_requires_grad(self.l2v, 'lora', True)
        
        if config.gradient_checkpointing:
            self.l2v.gradient_checkpointing_enable()

    
    def _init_l2v(self, config, device):
        assert config.bit in [4, 8, 16, 32], colorstr(f'Bits must be in [4, 8, 16, 32], but got {config.bit}..')
        
        if config.bit == 8:
            load_in_8bit, load_in_4bit = True, False
        elif config.bit == 4:
            load_in_8bit, load_in_4bit = False, True
        else:
            load_in_8bit, load_in_4bit = False, False

        if config.model.lower() == 'llama3':
            model = LLM2Vec.from_pretrained(
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
                device_map=device,
                use_cache=True,
                low_cpu_mem_usage=True,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                torch_dtype=torch.bfloat16 if config.bit == 16 else torch.float32,
                pooling_mode=self.pooling_mode,
                max_length=self.max_length
            )
        elif config.model.lower() == 'mistral':
            raise NotImplementedError
        
        return model, model.tokenizer
    
    
    @staticmethod
    def _chage_requires_grad(modules, target_module: str, grad_true: bool=True):
        for name, param in modules.named_parameters():
            if target_module in name:
                param.requires_grad = grad_true


    def forward(self, features):
        outputs = self.l2v(features)
        outputs = self.fc_layer(outputs)
        return outputs



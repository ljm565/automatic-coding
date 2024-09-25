import numpy as np
import importlib.metadata
from packaging import version
from transformers.utils.import_utils import _is_package_available
from transformers import (
    AutoModel,
    AutoConfig,
    PretrainedConfig,
    AutoTokenizer,
    LlamaConfig,
    MistralConfig,
    GemmaConfig,
    Qwen2Config,
)

import torch
from torch import Tensor

from utils import LOGGER, colorstr



def is_transformers_attn_greater_or_equal_4_38():
    if not _is_package_available("transformers"):
        return False

    return version.parse(importlib.metadata.version("transformers")) >= version.parse(
        "4.38.0"
    )


def is_transformers_attn_greater_or_equal_4_40():
    if not _is_package_available("transformers"):
        return False

    return version.parse(importlib.metadata.version("transformers")) >= version.parse(
        "4.40.0"
    )


def to_device(x, device):
    if isinstance(x, Tensor):
        return x.to(device)
    else:
        return {k: to_device(v, device) for k, v in x.items()}


def _convert_to_str(instruction, text, tokenizer, max_length):
    tokenized_q = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    tokenized_q_length = len(tokenized_q["input_ids"][0])

    while tokenized_q_length > max_length:
        reduction_ratio = max_length / tokenized_q_length
        reduced_length = int(len(text.split()) * reduction_ratio)
        text = " ".join(text.split()[:reduced_length])
        tokenized_q = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        tokenized_q_length = len(tokenized_q["input_ids"][0])

    return (
        f"{instruction.strip()} !@#$%^&*(){text}"
        if instruction
        else f"!@#$%^&*(){text}"
    )


def tokenize(texts, tokenizer, max_length, truncation=False, padding=True):
        texts_2 = []
        original_texts = []
        for text in texts:
            t = text.split("!@#$%^&*()")
            texts_2.append(t[1] if len(t) > 1 else "")
            original_texts.append("".join(t))

        original = tokenizer(
            original_texts,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        embed_mask = None
        for t_i, t in enumerate(texts_2):
            ids = tokenizer(
                [t],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            if embed_mask is None:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(
                        len(ids["input_ids"][0])
                    )
                embed_mask = e_m.unsqueeze(0)
            else:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(
                        len(ids["input_ids"][0])
                    )
                embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)

        original["embed_mask"] = embed_mask
        return original


def prepare_for_tokenization(text, l2v_config_instance, pooling_mode):
    if l2v_config_instance._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
        text = (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            + text.strip()
            + "<|eot_id|>"
        )
        return text
    if l2v_config_instance._name_or_path in [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
    ]:
        text = "[INST] " + text.strip() + " [/INST]"
    if l2v_config_instance._name_or_path in [
        "google/gemma-2-9b-it",
    ]:
        text = "<bos><start_of_turn>user\n" + text.strip() + "<end_of_turn>"
    if l2v_config_instance._name_or_path in [
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
    ]:
        text = "<|im_start|>user\n" + text.strip() + "<|im_end|>"
    if pooling_mode == "eos_token":
        if l2v_config_instance._name_or_path == "meta-llama/Meta-Llama-3-8B":
            text = text.strip() + "<|end_of_text|>"
        elif isinstance(l2v_config_instance, LlamaConfig) or isinstance(
            l2v_config_instance, MistralConfig
        ):
            text = text.strip() + " </s>"
        elif isinstance(l2v_config_instance, GemmaConfig):
            text = text.strip() + "<eos>"
        elif isinstance(l2v_config_instance, Qwen2Config):
            text = text.strip() + "<|endoftext|>"
    return text


def delete_all_zero_cols(pred: np.ndarray, gt: np.ndarray):
    zero_columns = np.where(np.all(gt == 0, axis=0))[0]
    
    # Delete all zero columns because of sklearn's metric error.
    # Please refer to CAML paper.
    gt = np.delete(gt, zero_columns, axis=1)
    pred = np.delete(pred, zero_columns, axis=1)

    assert gt.shape == pred.shape, 'GT and prediction size mismatch'
    return pred, gt


def print_samples(pred: np.ndarray, gt: np.ndarray, threshold: int):
    pred = np.where(pred >= threshold, 1, 0)

    # One-hot to indices
    pred = np.nonzero(pred)[0].tolist()
    gt = np.nonzero(gt)[0].tolist()

    LOGGER.info('\n' + '-'*100)
    LOGGER.info(colorstr(f'GT        : {gt}'))
    LOGGER.info(colorstr(f'Prediction: {pred}'))
    LOGGER.info('-'*100 + '\n')


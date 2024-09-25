import os
import random
import numpy as np
from tqdm import tqdm
from typing import List

import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils import LOGGER, colorstr
from utils.preprocess import get_hadm_dict, get_noteevent_dict
from utils.func_utils import _convert_to_str, tokenize, prepare_for_tokenization



def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_preprocessing(config):
    noteevent_path = os.path.join(config.mimic3_data_path, 'NOTEEVENTS.csv')
    hadm_paths = [os.path.join(config.mimic3_data_path, data) for data in ['DIAGNOSES_ICD.csv', 'PROCEDURES_ICD.csv']]
    code_paths = [os.path.join(config.mimic3_data_path, data) for data in ['D_ICD_DIAGNOSES.csv', 'D_ICD_PROCEDURES.csv']]
    columns = ['ICD9_CODE', 'ICD9_CODE']

    # Label and note event data dictionary
    labels, code_dict = get_hadm_dict(hadm_paths, code_paths, columns)
    noteevents = get_noteevent_dict(noteevent_path, config.chunk_size)

    # Remain only the necessary labels
    if config.label_pruning:
        idx2code = {v: k for k, v in code_dict.items()}
        labels = {k: [idx2code[e] for e in v] for k, v in labels.items()}
        used_code = set()
        for v in labels.values():
            used_code.update(v)
        
        # Make new code dictionary and update labels based on the new code dictionary
        code_dict = {code: i for i, code in enumerate(list(used_code))}
        labels = {k: [code_dict[e] for e in v] for k, v in labels.items()}
    
    config.n_labels = len(code_dict)
    return {'noteevents': noteevents, 'labels': labels, 'code_dict': code_dict}


def text_tokenize(sentence:str, l2v_config_instance, tokenizer, max_length, pooling_mode, truncation=False, padding=True):
    sentences = [[""] + [sentence]]

    concatenated_input_texts = []
    for sentence in sentences:
        assert isinstance(sentence[0], str)
        assert isinstance(sentence[1], str)
        concatenated_input_texts.append(
            _convert_to_str(sentence[0], sentence[1], tokenizer, max_length)
        )
    sentences = concatenated_input_texts

    tokens = tokenize(
        [prepare_for_tokenization(sentence, l2v_config_instance, pooling_mode) for sentence in sentences],
        tokenizer,
        max_length,
        truncation,
        padding
    )

    return tokens



class DLoader(Dataset):
    def __init__(self, data, l2v_model_config, tokenizer, config):
        self.data = self.preprocess(data)
        self.tokenizer = tokenizer
        self.max_length = config.max_len
        self.pooling_mode = config.pooling_mode
        self.l2v_model_config = l2v_model_config
        self.n_labels = config.n_labels
        self.length = len(self.data)


    def preprocess(self, data):
        collected_data = list()
        noteevents, labels = data['noteevents'], data['labels']
        for hadm, text in tqdm(noteevents.items(), desc="Collecting pre-processed data.."):
            try:
                text = '\n\n'.join(text)
                collected_data.append((text, labels[hadm]))
            except KeyError:
                LOGGER.warning(colorstr('yellow', f'HADM "{hadm}"\'s ICD code does not exist in the ICD code dictionary.'))
                continue
        return collected_data
    

    def tokenize(self, text):
        inputs = text_tokenize(
            text, 
            self.l2v_model_config,
            self.tokenizer, 
            self.max_length, 
            self.pooling_mode,
            padding='max_length',
            truncation=True
        )
        return inputs


    @staticmethod
    def one_hot_encoding(n_labels: int, multi_labels: List[int]):
        one_hot_matrix = torch.zeros(n_labels)
        one_hot_matrix[multi_labels] = 1
        return one_hot_matrix
    

    @staticmethod
    def squeeze(inputs):
        for k, v in inputs.items():
            if isinstance(v, Tensor):
                inputs[k] = v.squeeze(0)
        return inputs


    def __getitem__(self, idx):
        inputs = self.squeeze(self.tokenize(self.data[idx][0]))
        label = self.one_hot_encoding(self.n_labels, self.data[idx][1])
        return inputs, label


    def __len__(self):
        return self.length
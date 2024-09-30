import os
from typing import List
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, distributed

from models import AutomaticCodingModel
from utils import RANK, colorstr
from utils.data_utils import DLoader, seed_worker

PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders



def get_model(config, device):
    model = AutomaticCodingModel(config, device).to(device)
    tokenizer = model.tokenizer
    return model, tokenizer


def build_dataset(config, preprocessed_data, l2v_model_config, tokenizer, modes):
    def _split_dataset(preprocessed_data, split_digit: List[str]):
        trainset = {'noteevents': {}, 'labels': {}, 'code_dict': deepcopy(preprocessed_data['code_dict'])}
        valset = {'noteevents': {}, 'labels': {}, 'code_dict': deepcopy(preprocessed_data['code_dict'])}

        for key in ['noteevents', 'labels']:
            for hadm, v in preprocessed_data[key].items():
                if hadm[-1] in split_digit:
                    valset[key][hadm] = deepcopy(v)
                else:
                    trainset[key][hadm] = deepcopy(v)
        
        return trainset, valset
    
    assert all(key in preprocessed_data for key in ['labels', 'noteevents', 'code_dict']), colorstr('red', f'Pre-processed data have to contain "labels", "noteevents", and "code_dict"..')
    
    trainset, valset = _split_dataset(preprocessed_data, ['8','9'])
    split_dataset = {'train': trainset, 'validation': valset}
    dataset_dict = {s: DLoader(d, l2v_model_config, tokenizer, config) for s, d in split_dataset.items() if s in modes}
    
    return dataset_dict


def build_dataloader(dataset, batch, workers, shuffle=True, is_ddp=False):
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if not is_ddp else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return DataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=getattr(dataset, 'collate_fn', None),
                              worker_init_fn=seed_worker,
                              generator=generator)


def get_data_loader(config, preprocessed_data, l2v_model_config, tokenizer, modes, is_ddp=False):
    datasets = build_dataset(config, preprocessed_data, l2v_model_config, tokenizer, modes)
    dataloaders = {m: build_dataloader(datasets[m], 
                                       config.batch_size, 
                                       config.workers, 
                                       shuffle=(m == 'train'), 
                                       is_ddp=is_ddp) for m in modes if m in datasets}

    return dataloaders
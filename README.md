# ICD-9 Automatic Coding (Multi-label Classification) with LLM2Vec
한국어 버전의 설명은 [여기](./docs/README_ko.md)를 참고하시기 바랍니다.

## Introduction
We aim to predict a patient's ICD-9 disease codes based on the text-based clinical records in the NOTEEVENTS section of the [MIMIC-III](https://mimic.mit.edu/docs/iii/) dataset.
Using data from DIAGNOSES_ICD.csv and PROCEDURES_ICD.csv in [MIMIC-III](https://mimic.mit.edu/docs/iii/), we can extract the patient’s HADM number and ICD-9 codes.
Patients whose HADM numbers end in 8 or 9 are used as the validation set.
There are more than 18,000 ICD-9 codes, and patients can have multiple ICD-9 codes (Multi-label). 
While encoder-only models like BERT can be used for multi-label prediction, in this case, we will use the [LLM2Vec](https://github.com/McGill-NLP/llm2vec) model, which is based on the LLaMA3 model but with the causal mask removed and pre-trained as an encoder. 
The data used in this experiment contains sensitive medical information and is not publicly accessible, so the data will not be disclosed.
<br><br><br>

## Supported Models
### [Pre-trained LLM2Vec](https://github.com/McGill-NLP/llm2vec)
* LLaMA3-based LLM2Vec model.
* Mistral-based LLM2Vec model.
<br><br><br>

## Base Dataset
* [MIMIC-III](https://mimic.mit.edu/docs/iii/)
    * D_ICD_DIAGNOSES.csv
    * D_ICD_PROCEDURES.csv
    * DIAGNOSES_ICD.csv
    * PROCEDURES_ICD.csv
    * NOTEEVENTS.csv
<br><br><br>

## Supported Devices
* CPU, GPU, multi-GPU (DDP), MPS (for Mac and torch>=1.12.0)
<br><br><br>

## Quick Start
```bash
python3 src/run/train.py --config config/config.yaml --mode train
```
<br><br>


## Project Tree
This repository is structured as follows.
```
├── configs                           <- Folder for storing config files
│   └── *.yaml
│
└── src      
    ├── models
    |   ├── llm2vec                   <- LLaMA3 and Mistral-based LLM2Vec model folder
    |   └── auto_coding_model.py      <- Final automatic coding model
    |
    ├── run                   
    |   ├── train.py                  <- Training execution file
    |   └── validation.py             <- Trained model evaulation execution file
    |
    ├── tools                   
    |   ├── early_stopper.py          <- Early stopper class file
    |   ├── model_manager.py          
    |   └── training_logger.py        <- Training logger class file
    |
    ├── trainer                 
    |   ├── build.py                  <- Codes for initializing dataset, dataloader, etc.
    |   └── trainer.py                <- Class for training, evaluating, and calculating accuracy
    |
    └── uitls                   
        ├── __init__.py               <- File for initializing the logger, versioning, etc.
        ├── data_utils.py             <- File defining the dataset's dataloader
        ├── filesys_utils.py       
        ├── func_utils.py
        ├── model_utils.py       
        ├── preprocess.py             <- MIMIC-III pre-processing file       
        └── training_utils.py     
```
<br><br>

## Tutorials & Documentations
Please follow the steps below to train an automatic coding (multi-label classification) model.
1. [Getting Started](./docs/1_getting_started.md)
2. [Data Preparation](./docs/2_data_preparation.md)
3. [Training](./docs/3_trainig.md)
4. ETC
   * [Evaluation](./docs/4_model_evaluation.md)

<br><br><br>


## Training Results
### LLaMA3-based LLM2Vec Results
* AUCROC Macro: 0.9057
* AUCROC Micro: 0.9808
* AUCPRC Macro: 0.2073
* AUCPRC Micro: 0.4741


## Acknowledgement
* [LLM2Vec](https://github.com/McGill-NLP/llm2vec)
# ICD-9 Automatic Coding (Multi-label Classification) with LLM2Vec

## Introduction
[MIMIC-III](https://mimic.mit.edu/docs/iii/)의 NOTEEVENT의 텍스트 진료 기록을 바탕으로 환자의 ICD-9 질병 코드를 예측합니다.
[MIMIC-III](https://mimic.mit.edu/docs/iii/)의 DIAGNOSES_ICD.csv와 PROCEDURES_ICD.csv의 데이터에 있는 환자의 HADM과 ICD-9 코드를 추출할 수 있습니다.
이때 추출한 환자의 HADM 번호 중 8, 9로 끝나는 환자는 validation set으로 사용합니다.
ICD-9 코드는 18,000개 이상 존재하며 이때 환자는 여러개의 ICD-9 코드를 가질 수 있습니다(Multi-label).
Multi-label을 예측하는 모델은 BERT 처럼 encoder-only 모델을 사용할 수 있지만, 여기서는 LLM의 causal mask를 없애고 encoder로 pre-training한 LLaMA3 기반의 [LLM2Vec](https://github.com/McGill-NLP/llm2vec) 모델을 사용합니다.
본 실험에서 사용한 데이터는 민감한 의료 데이터이며 접근 권한이 없으면 데이터를 볼 수 없기 때문에 데이터는 공개하지 않겠습니다.
<br><br><br>

## Supported Models
### [Pre-trained LLM2Vec](https://github.com/McGill-NLP/llm2vec)
* LLaMA3 기반의 LLM2Vec 모델.
* Mistral 기반의 LLM2Vec 모델.
<br><br><br>

## 사용 데이터
* [MIMIC-III](https://mimic.mit.edu/docs/iii/)
    * D_ICD_DIAGNOSES.csv
    * D_ICD_PROCEDURES.csv
    * DIAGNOSES_ICD.csv
    * PROCEDURES_ICD.csv
    * NOTEEVENTS.csv
<br><br><br>

## Supported Devices
* CPU, GPU, multi-GPU (DDP), MPS (for Mac and torch==2.2.2)
<br><br><br>

## Quick Start
```bash
python3 src/run/train.py --config config/config.yaml --mode train
```
<br><br>


## Project Tree
본 레포지토리는 아래와 같은 구조로 구성됩니다.
```
├── configs                           <- Config 파일들을 저장하는 폴더
│   └── *.yaml
│
└── src      
    ├── models
    |   ├── llm2vec                   <- LLaMA3, Mistral 기반의 LLM2Vec 모델 폴더
    |   └── auto_coding_model.py      <- 최종 automatic coding을 위한 모델
    |
    ├── run                   
    |   ├── train.py                  <- 학습 실행 파일
    |   └── validation.py             <- 학습된 모델 평가 실행 파일
    |
    ├── tools                   
    |   ├── early_stopper.py          <- Early stopper class 파일
    |   ├── model_manager.py          
    |   └── training_logger.py        <- Training logger class 파일
    |
    ├── trainer                 
    |   ├── build.py                  <- Dataset, dataloader 등을 정의하는 파일
    |   └── trainer.py                <- 학습, 평가 등을 수행하는 class 파일
    |
    └── uitls                   
        ├── __init__.py               <- Logger, 버전 등을 초기화 하는 파일
        ├── data_utils.py             <- Dataloader 정의 파일
        ├── filesys_utils.py       
        ├── func_utils.py       
        ├── model_utils.py       
        ├── preprocess.py             <- MIMIC-III pre-processing 파일
        └── training_utils.py     
```
<br><br>

## Tutorials & Documentations
Automatic coding (multi-label classification) 모델 학습을 위해서 다음 과정을 따라주시기 바랍니다.
1. [Getting Started](./1_getting_started_ko.md)
2. [Data Preparation](./2_data_preparation_ko.md)
3. [Training](./3_trainig_ko.md)
4. ETC
   * [Evaluation](./4_model_evaluation_ko.md)

<br><br><br>


## Training Results
### LLaMA3-based LLM2Vec Results
* AUCROC Macro: 0.9057
* AUCROC Micro: 0.9808
* AUCPRC Macro: 0.2073
* AUCPRC Micro: 0.4741
<br><br><br>

## Acknowledgement
* [LLM2Vec](https://github.com/McGill-NLP/llm2vec)
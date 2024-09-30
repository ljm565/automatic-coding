# Data Preparation
여기서는 automatic coding 모델 훈련을 위해 [MIMIC-III](https://mimic.mit.edu/docs/iii/) 데이터셋을 사용합니다.
본 실험에서 사용한 데이터는 민감한 의료 데이터이며 접근 권한이 없으면 데이터를 볼 수 없기 때문에 데이터는 공개하지 않겠습니다.
만약 아래에서 설명할 config 파일의 data 경로에, 액세스 권한이 있거나 이미 데이터를 보유하고 있는 경우, 아래 목록의 데이터를 넣으면 됩니다.
* D_ICD_DIAGNOSES.csv
* D_ICD_PROCEDURES.csv
* DIAGNOSES_ICD.csv
* PROCEDURES_ICD.csv
* NOTEEVENTS.csv

### 1. MIMIC-III
위의 데이터 세트 목록을 포함해야 하는 `mimic3_data_path`를 설정해야 합니다.
```yaml
mimic3_data_path: data/
```
<br>

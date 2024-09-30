# Data Preparation
Herein, we will use [MIMIC-III](https://mimic.mit.edu/docs/iii/) dataset for the automatic coding model training.
The data used in this experiment contains sensitive medical information and is not publicly accessible, so the data will not be disclosed.
If you have access permissions or already possess the data, simply place the data from the following list into the data path specified in the configuration file that will be explained next:
* D_ICD_DIAGNOSES.csv
* D_ICD_PROCEDURES.csv
* DIAGNOSES_ICD.csv
* PROCEDURES_ICD.csv
* NOTEEVENTS.csv

### 1. MIMIC-III
You need to set `mimic3_data_path` which should contain the list of datasets above.
```yaml
mimic3_data_path: data/
```
<br>

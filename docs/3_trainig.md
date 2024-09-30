# Training an Automatic Coding Model
Here, we provide guides for training an automatic coding model.

### 1. Configuration Preparation
To train a model, you need to create a configuration.
Detailed descriptions and examples of the configuration options are as follows.

```yaml
# base
seed: 0
deterministic: True

# environment config
device: [0]     # examples: [0], [0,1], [1,2,3], cpu, mps... 

# project config
project: outputs/mimic3
name: automatic_coding

# model config
model: llama3                   # [llama3, mistral]
bit: 8
max_len: 1024
pooling_mode: mean              # [mean, weighted_mean, bos_token, eos_token]
lora_activate: True             # If True, LoRA requires_grad will be set to True.

# data config
workers: 0                      # Don't worry to set worker. The number of workers will be set automatically according to the batch size.
mimic3_data_path: data/
chunk_size: 100000              # Pandas data chunck size during reading NOTEEVENT data.
label_pruning: False            # If True, only the required ICD codes will be used, otherwise all codes will be used.

# train config
batch_size: 8
steps: 60000
warmup_steps: 400
lr0: 1e-4
lrf: 0.01                             # last_lr = lr0 * lrf
scheduler_type: 'cosine'              # ['linear', 'cosine']
pos_weight: 5                         # Positive label weight of the loss function.
gradient_checkpointing: True
patience: 5                          # Early stopping epochs.
prediction_print_n: 3                # Number of examples to show during inference.
positive_threshold: 0.5

# logging config
common: ['train_loss', 'validation_loss', 'lr']
metrics: ['aucroc_macro', 'aucroc_micro', 'aucprc_macro', 'aucprc_micro']   # You can add more metrics after implements metric validation codes.
```


### 2. Training
#### 2.1 Arguments
There are several arguments for running `src/run/train.py`:
* [`-c`, `--config`]: Path to the config file for training.
* [`-m`, `--mode`]: Choose one of [`train`, `resume`].
* [`-r`, `--resume_model_dir`]: Path to the model directory when the mode is resume. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to resume.
* [`-l`, `--load_model_type`]: Choose one of [`metric`, `loss`, `last`].
    * `metric` (default): Resume the model with the best validation set's metrics (e.g. AUCROC, AUCPRC).
    * `loss`: Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-p`, `--port`]: (default: `10001`) NCCL port for DDP training.


#### 2.2 Command
`src/run/train.py` file is used to train the model with the following command:
```bash
# training from scratch
python3 src/run/train.py --config configs/config.yaml --mode train

# training from resumed model
python3 src/run/train.py --config config/config.yaml --mode resume --resume_model_dir ${project}/${name}
```

When training started, the learning rate curve will be saved in `${project}/${name}/vis_outputs/lr_schedule.png` automatically based on the values set in `config/config.yaml`.
When the model training is complete, the checkpoint is saved in `${project}/${name}/weights` and the training config is saved at `${project}/${name}/args.yaml`.
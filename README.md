# data2vec-KWS
This repository contains code for applying [Data2Vec](https://arxiv.org/abs/2202.03555) to pretrain the [KWT model by Axel Berg](https://arxiv.org/abs/2104.00769) as described in [Improving Label-Deficient Keyword Spotting Through Self-Supervised Pretraining](https://arxiv.org/abs/2210.01703).
The goal was to improve keyword spotting when only a small amount of labelled data is available.

Experiments are carried out on a reduced labelled setup of the Google Speech Commands V2 data set.
In the reduced setup 80% of the training set is used for unlabelled pretraining using Data2Vec, and only 20% for labelled training. 

The code was developed as part of a Master project at Aalborg University, Aalborg, Denmark.
Much of the code is based on a PyTorch implementation of the KWT model which can be found [here](https://github.com/ID56/Torch-KWT/blob/main/models/kwt.py).

Additionally, the Data2Vec module takes inspiration from another effort to implement Data2Vec in PyTorch found [here](https://github.com/arxyzan/data2vec-pytorch), 
as well as the published code in the [FAIRSEQ python library](https://github.com/facebookresearch/fairseq).

## Setup
The codebase is implemented in Python and tested for Python 3.8 and 3.9.
It is reccomended to first setup a virual environment, e.g. using venv (using conda is also a possibility):

```shell
python -m venv venv
```


The necessary Python packages to run the code is installed by running:

```shell
pip install -r requirements.txt
```

To download the Google Speech Commands V2 data set run the command:

```bash
bash download_gspeech_v2.sh <path/to/dataset/root>
```

For example:

```bash
bash download_gspeech_v2.sh speech_commands_v0.02
```

The text files with the data splits of the reduced labelled setup can be generated by running:

```bash
python make_data_list.py --pretrain <amount_of_train_set_for_pretrain> -v <path/to/validation_list.txt> -t <path/to/testing_list.txt> -d <path/to/dataset/root> -o <output dir>
```

For example:
```bash
python make_data_list.py --pretrain 0.8 -v speech_commands_v0.02/validation_list.txt -t speech_commands_v0.02/testing_list.txt -d speech_commands_v0.02 -o speech_commands_v0.02/_generated
```

## Google Speech Commands
The Google Speech Commands V2 data set consists of 105 829 labelled keyword sequences of approximately 1 s.

The original train, validation, test splits are 80:10:10. 
For experiments 80% of the training set have been used for unlabelled pretraining and the last 20% for labelled training.
This yields the following splits:

| Split       | No. keyword examples |
|-------------|----------------------|
| Pretraining | 67 874               |
| Training    | 16 969               |
| Validation  |  9 981               |
| Testing     | 11 005               |


## Experiment configuration
The configurations for all experiments are stored in `.yaml` files, which contain everything from the data path to hyperparameters.
You will need to ensure that the paths in the configuration file matches your local data paths.

Configuration files for all experiments regarding the above-mentioned dataset are provided in the `KWT_configs` and `data2vec/data2vec_configs` folders.
This includes configuration for KWT model baselines on the reduced Speech Commands training set, Data2Vec pretraining configurations for Speech Commands pretratining set and finetuning on the reduced Speech Commands training set.

Config files for baseline and finetuning containing "mean" in the filename uses the mean of the tranformer outputs as input for the classification head,
while the ones that don't use a CLS token as in the original KWT model. 
The best results are obtained using mean.

## Baseline KWT results
To produce the baseline results without pretraining the following command is run:

```bash
python train.py --conf KWT_configs/<name_of_config_file>.yaml
```

For example:
```bash
python train.py --conf KWT_configs/kwt1_baseline_mean_config.yaml
```

Config files for baseline experiments are found in the `KWT_configs` folder.


## Data2Vec pretraining
To pretrain a KWT model the following command is run:

```bash
python train_data2vec.py --conf data2vec/data2vec_configs/<name_of_config_file>.yaml
```

For example:

```bash
python train_data2vec.py --conf data2vec/data2vec_configs/kwt1_data2vec_config.yaml
```

Data2vec config files are found in the `data2vec/data2vec_configs/` folder.

After pretraining the full Data2Vec model is saved to the experiment folder, as well as only the transformer encoder part.
For finetuning only the transformer encoder is reused.

## Finetuning pretrained models
To finetune a pretrained model the same setup as the baseline is used, however, a pretrained checkpoint is first loaded for the encoder part of the model.
A pretrained kwt1 model checkpoint is found in the repository root, called ```kwt1_pretrained.pth```.

The path to the pretrained model is provided through an argument:

```shell
python train.py --conf KWT_configs/<name_of_finetune_config_file>.yaml --ckpt <path_to_checkpoint.pth>
```

For example:
```bash
python train.py --conf KWT_configs/kwt1_finetune_mean_config.yaml --ckpt runs/kwt1_data2vec/best_encoder.pth
```

Finetuning config files are found in the `KWT_configs` folder.

# Results
The following table is a summary of accuracies obtained for the three KWT models. 
Baseline is the performance without pretraining using the reduced training set with only 20% of the labelled data.
SC denotes finetuning on the reduced Speech Commands training set after Data2Vec pretraining using Speech Commands pretraining set.
SC-FE denotes fine-tuning with the encoder weights frozen, such that only the linear classification head is trained (Linear Probing).
Additionally, accuracies for Librispeech pretraining. 
LS denotes finetuning on the reduced Speech Commands training set after Data2Vec pretraining using the Librispeech 100-hour clean training set. 


| Model 	| Baseline 	| Data2Vec-SC 	| Data2Vec-SC-FE 	| Data2Vec-LS 	|
|-------	|----------	|-------------	|----------------	|-------------	|
| KWT-1 	| 0.8622   	| 0.9294      	| 0.4292         	| 0.9436      	|
| KWT-2 	| 0.8575   	| 0.9507      	| 0.4974         	| 0.9447      	|
| KWT-3 	| 0.8398   	| 0.9529      	| 0.4960         	| 0.9458      	|

We also tested the performance benefit when all the labelled data is available.
Full indicates training on the full original Speech Commands V2 training set without Data2Vec pretraining.
LS-Pretrain + Full denotes a model fine-tuned on the full original training set after pretraining on librispeech.

| Model 	| Full         	| Full + LS pretrain 	|
|-------	|--------------	|--------------------	|
| KWT-1 	| 0.9638       	| 0.9713             	|
| KWT-2 	| 0.9498       	| 0.9716             	|
| KWT-3 	| 0.9079 	      | 0.9716             	|

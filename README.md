# data2vec-KWS
This repository contains code for applying [Data2Vec](https://arxiv.org/abs/2202.03555) to pretrain the [KWT model by Axel Berg](https://arxiv.org/abs/2104.00769).
The goal was to improve keyword spotting when only a small amount of labelled data is available.

Experiments are carried out on a reduced labelled setup of the Google Speech Commands V2 data set.
In the reduced setup 80% of the training set is used for unlabelled pretraining using Data2Vec, and only 20% for labelled training. 

The code was developed as part of a Master project at Aalborg University, Aalborg, Denmark.
Much of the code is based on a PyTorch implementation of the KWT model which can be found [here](https://github.com/ID56/Torch-KWT/blob/main/models/kwt.py).

Additionally, the Data2Vec module takes inspiration from another effort to implement Data2Vec in PyTorch found [here](https://github.com/arxyzan/data2vec-pytorch), 
as well as the published code in the [FAIRSEQ python library](https://github.com/facebookresearch/fairseq).

## Setup
The codebase is implemented in Python and tested for Python 3.8 and 3.9.
The necessary Python packages to run the code is installed by running:

```shell
pip install -r requirements.txt
```

To download the Google Speech Commands V2 data set run the command:

```shell
sh download_gspeech_v2.sh
```

The text files with the data splits of the reduced labelled setup can be generated by running:

```shell
python make_data_list.py --pretrain <amount_of_train_set_for_pretrain> -v <path/to/validation_list.txt> -t <path/to/testing_list.txt> -d <path/to/dataset/root> -o <output dir>
```

For example:
```shell
python make_data_list.py --pretrain 0.8 -v speech_commands_v0.02/validation_list.txt -t speech_commands_v0.02/testing_list.txt -d speech_commands_v0.02 -o data_split_lists
```

## Google Speech Commands
The Google Speech Commands V2 data set consists of 105 829 labelled keyword sequences of approximately 1 s.

The original train, validation, test splits are 80:10:10. 
For experiments 80% of the training set have been used for unlabelled pretraining.
This yields the following splits:

| Split       | No. keyword examples |
|-------------|----------------------|
| Pretraining | 67 731               |
| Training    | 16 932               |
| Validation  | 10 583               |
| Testing     | 10 583               |


## Baseline KWT results
To produce the baseline results the following command is run 

```shell
python train.py --conf KWT_configs/<name_of_config_file>.yaml
```

An example of a config file is found in the `KWT_configs` folder

## Data2Vec pretraining
To pretrain a KWT model the following command is run 

```shell
python train_data2vec.py --conf data2vec/data2vec_configs/<name_of_config_file>.yaml
```

An example of a config file is found in the `data2vec/data2vec_configs/` folder


## Finetuning pretrained models
To finetune a pretrained model the same setup as the baseline is used, however, a pretrained checkpoint is first loaded.
The path to the pretrained model is provided through an input argument:

```shell
python train.py --conf KWT_configs/<name_of_finetune_config_file>.yaml --ckpt <path_to_checkpoint.pth>
```

An example of a finetuning config file is found in the `KWT_configs` folder.
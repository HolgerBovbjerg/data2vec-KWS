"""
Module for creating Google Speech Commands dataset
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import functools
from itertools import compress
import librosa
import glob
import os
from tqdm import tqdm
import multiprocessing as mp
import json

from utils.augment import time_shift, resample, spec_augment
from audiomentations import AddBackgroundNoise


def get_train_val_test_split(root: str, train_file: str, val_file: str, test_file: str,
                             pretrain_fraction: float = None):
    """Creates train, val, and test split according to provided val and test files.

    Args:
        root (str): Path to base directory of the dataset.
        val_file (str): Path to file containing list of validation data files.
        test_file (str): Path to file containing list of test data files.
        pretrain_fraction (float): Value between 0 and 1, depending on how big pretrain set should be.

    Returns:
        train_list (list): List of paths to training data items.
        val_list (list): List of paths to validation data items.
        test_list (list): List of paths to test data items.
    """

    with open(train_file) as file:
        train_data = json.load(file)
        train_duration = [element["duration"] for element in train_data]
        train_list = [root + element["audio_file_path"] for element in train_data]
        train_labels = [element["is_hotword"] for element in train_data]
        train_list = list(compress(train_list, [duration <= 5 for duration in train_duration]))
        train_labels = list(compress(train_labels, [duration <= 5 for duration in train_labels]))

    with open(val_file) as file:
        val_data = json.load(file)
        val_duration = [element["duration"] for element in val_data]
        val_list = [root + element["audio_file_path"] for element in val_data]
        val_labels = [element["is_hotword"] for element in val_data]
        val_list = list(compress(val_list, [duration <= 5 for duration in val_duration]))
        val_labels = list(compress(val_labels, [duration <= 5 for duration in val_labels]))

    with open(test_file) as file:
        test_data = json.load(file)
        test_duration = [element["duration"] for element in test_data]
        test_list = [root + element["audio_file_path"] for element in test_data]
        test_labels = [element["is_hotword"] for element in test_data]
        test_list = list(compress(test_list, [duration <= 5 for duration in test_duration]))
        test_labels = list(compress(test_labels, [duration <= 5 for duration in test_duration]))

    ###################
    # Pretrain Split
    ###################

    pretrain_list = []
    pretrain_labels = []
    if pretrain_fraction is not None:
        pretrain_len = round(len(train_list) * pretrain_fraction)
        pretrain_list = train_list[:pretrain_len]
        pretrain_labels = train_labels[:pretrain_len]
        train_list = train_list[pretrain_len:]
        train_labels = train_labels[pretrain_len:]

    print(f"Number of pretraining samples: {len(pretrain_list)}")
    print(f"Number of training samples: {len(train_list)}")
    print(f"Number of validation samples: {len(val_list)}")
    print(f"Number of test samples: {len(test_list)}")

    pretrain = {"data": pretrain_list,
                "labels": pretrain_labels}
    train = {"data": train_list,
             "labels": train_labels}
    val = {"data": val_list,
           "labels": val_labels}
    test = {"data": test_list,
            "labels": test_labels}

    return pretrain, train, val, test


class SnipsDataset(Dataset):
    """Dataset wrapper for Snips Dataset."""

    def __init__(self, data_list: list, audio_settings: dict, aug_settings: dict = None,
                 cache: int = 0, label_list=None):
        super().__init__()

        self.audio_settings = audio_settings
        self.aug_settings = aug_settings
        self.cache = cache
        if cache:
            print("Caching dataset into memory.")
            self.data_list = init_cache(data_list, audio_settings["sr"], cache, audio_settings)
        else:
            self.data_list = data_list

        # labels: if no label map is provided, will not load labels. (Use for inference)
        if label_list is not None:
            self.label_list = [int(label) for label in label_list]
        else:
            self.label_list = None

        if aug_settings is not None and "bg_noise" in self.aug_settings:
            self.bg_adder = AddBackgroundNoise(sounds_path=aug_settings["bg_noise"]["bg_folder"])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.cache:
            x = self.data_list[idx]
        else:
            x = librosa.load(self.data_list[idx], self.audio_settings["sr"])[0]

        x = self.transform(x)

        if self.label_list is not None:
            label = torch.tensor(self.label_list[idx])
            return x, label
        else:
            return x

    def transform(self, x):
        """Applies necessary preprocessing to audio.

        Args:
            x (np.ndarray) - Input waveform; array of shape (n_samples, ).
        
        Returns:
            x (torch.FloatTensor) - MFCC matrix of shape (n_mfcc, T).
        """

        sr = self.audio_settings["sr"]

        ###################
        # Waveform 
        ###################

        if self.cache < 2:

            if self.aug_settings is not None:
                if "bg_noise" in self.aug_settings:
                    x = self.bg_adder(samples=x, sample_rate=sr)

                if "time_shift" in self.aug_settings:
                    x = time_shift(x, sr, **self.aug_settings["time_shift"])

                if "resample" in self.aug_settings:
                    x, _ = resample(x, sr, **self.aug_settings["resample"])

            x = librosa.util.fix_length(x, size=5 * sr)

            ###################
            # Spectrogram
            ###################

            x = librosa.feature.melspectrogram(y=x, **self.audio_settings)
            x = librosa.feature.mfcc(S=librosa.power_to_db(x), n_mfcc=self.audio_settings["n_mels"])

        if self.aug_settings is not None and "spec_aug" in self.aug_settings:
            x = spec_augment(x, **self.aug_settings["spec_aug"])

        x = torch.from_numpy(x).float().unsqueeze(0)
        return x


def cache_item_loader(path: str, sr: int, cache_level: int, audio_settings: dict) -> np.ndarray:
    x = librosa.load(path, 5 * sr)[0]
    if cache_level == 2:
        x = librosa.util.fix_length(x, 5 * sr)
        x = librosa.feature.melspectrogram(y=x, **audio_settings)
        x = librosa.feature.mfcc(S=librosa.power_to_db(x), n_mfcc=audio_settings["n_mels"])
    return x


def init_cache(data_list: list, sr: int, cache_level: int, audio_settings: dict, n_cache_workers: int = 4) -> list:
    """Loads entire dataset into memory for later use.

    Args:
        data_list (list): List of data items.
        sr (int): Sampling rate.
        cache_level (int): Cache levels, one of (1, 2), caching wavs and spectrograms respectively.
        n_cache_workers (int, optional): Number of workers. Defaults to 4.

    Returns:
        cache (list): List of data items.
    """

    cache = []

    loader_fn = functools.partial(cache_item_loader, sr=sr, cache_level=cache_level, audio_settings=audio_settings)

    pool = mp.Pool(n_cache_workers)

    for audio in tqdm(pool.imap(func=loader_fn, iterable=data_list), total=len(data_list)):
        cache.append(audio)

    pool.close()
    pool.join()

    return cache


def get_loader(data_list, label_list, config, train=True):
    """
    Creates dataloader for training, validation or testing
    :param data_list: Path to data
    :param label_list: Path to labels
    :param config: Configuration
    :param train: Specifies whether loader is used for training. If True data is shuffled.
    :return: PyTorch dataloader
    """

    dataset = SnipsDataset(
        data_list=data_list,
        label_list=label_list,
        audio_settings=config["hparams"]["audio"],
        aug_settings=config["hparams"]["augment"] if train else None,
        cache=config["exp"]["cache"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["hparams"]["batch_size"],
        num_workers=config["exp"]["n_workers"],
        pin_memory=config["exp"]["pin_memory"],
        shuffle=True if train else False
    )

    return dataloader

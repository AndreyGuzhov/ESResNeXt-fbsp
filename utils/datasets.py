import os
import warnings
import multiprocessing as mp

import msgpack_numpy as m

import numpy as np
import pandas as pd
import sklearn.model_selection as skms

import tqdm
import librosa

import torch.utils.data as td

from utils import transforms

from typing import Tuple
from typing import Optional


m.patch()


class ESC50(td.Dataset):

    def __init__(self,
                 root: str,
                 sample_rate: int = 22050,
                 train: bool = True,
                 fold: Optional[int] = None,
                 transform=None,
                 target_transform=None):

        super(ESC50, self).__init__()

        self.sample_rate = sample_rate

        meta = self.load_meta(os.path.join(root, 'meta', 'esc50.csv'))

        if fold is None:
            fold = 5

        self.folds_to_load = set(meta['fold'])

        if fold not in self.folds_to_load:
            raise ValueError(f'fold {fold} does not exist')

        self.train = train
        self.transform = transform

        if self.train:
            self.folds_to_load -= {fold}
        else:
            self.folds_to_load -= self.folds_to_load - {fold}

        self.data = dict()
        self.load_data(meta, os.path.join(root, 'audio'))
        self.indices = list(self.data.keys())

        self.target_transform = target_transform

    @staticmethod
    def load_meta(path_to_csv: str) -> pd.DataFrame:
        meta = pd.read_csv(path_to_csv)

        return meta

    @staticmethod
    def _load_worker(idx: int, filename: str, sample_rate: Optional[int] = None) -> Tuple[int, int, np.ndarray]:
        wav, sample_rate = librosa.load(filename, sr=sample_rate, mono=True)

        if wav.ndim == 1:
            wav = wav[:, np.newaxis]

        wav = wav.T * 32768.0

        return idx, sample_rate, wav.astype(np.float32)

    def load_data(self, meta: pd.DataFrame, base_path: str):
        items_to_load = dict()

        for idx, row in meta.iterrows():
            if row['fold'] in self.folds_to_load:
                items_to_load[idx] = os.path.join(base_path, row['filename']), self.sample_rate

        items_to_load = [(idx, path, sample_rate) for idx, (path, sample_rate) in items_to_load.items()]

        warnings.filterwarnings('ignore')
        with mp.Pool(processes=mp.cpu_count()) as pool:
            chunksize = int(np.ceil(len(items_to_load) / pool._processes)) or 1
            tqdm.tqdm.write(f'Loading {self.__class__.__name__} (train={self.train})')
            for idx, sample_rate, wav in pool.starmap(
                    func=self._load_worker,
                    iterable=items_to_load,
                    chunksize=chunksize
            ):
                row = meta.loc[idx]

                self.data[idx] = {
                    'audio': wav,
                    'sample_rate': sample_rate,
                    'target': row['target'],
                    'fold': row['fold'],
                    'esc10': row['esc10']
                }

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        if not (0 <= index < len(self)):
            raise IndexError

        audio: np.ndarray = self.data[self.indices[index]]['audio']
        target: int = self.data[self.indices[index]]['target']

        if self.transform is not None:
            audio = self.transform(audio)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return audio, target

    def __len__(self) -> int:
        return len(self.indices)


class UrbanSound8K(td.Dataset):

    def __init__(self,
                 root: str,
                 sample_rate: int = 22050,
                 train: bool = True,
                 fold: Optional[int] = None,
                 random_split_seed: Optional[int] = None,
                 mono: bool = False,
                 transform=None,
                 target_transform=None):

        super(UrbanSound8K, self).__init__()

        self.root = root
        self.sample_rate = sample_rate
        self.train = train

        if fold is None:
            fold = 1

        if not (1 <= fold <= 10):
            raise ValueError(f'Expected fold in range [1, 10], got {fold}')

        self.fold = fold
        self.folds_to_load = set(range(1, 11))

        if self.fold not in self.folds_to_load:
            raise ValueError(f'fold {fold} does not exist')

        if self.train:
            # if in training mode, keep all but test fold
            self.folds_to_load -= {self.fold}
        else:
            # if in evaluation mode, keep the test samples only
            self.folds_to_load -= self.folds_to_load - {self.fold}

        self.random_split_seed = random_split_seed
        self.mono = mono

        self.transform = transform
        self.target_transform = target_transform

        self.data = dict()
        self.indices = dict()
        self.load_data()

    @staticmethod
    def _load_worker(fn: str, path_to_file: str, sample_rate: int, mono: bool = False) -> Tuple[str, int, np.ndarray]:
        wav, sample_rate = librosa.load(path_to_file, sr=sample_rate, mono=mono)

        if wav.ndim == 1:
            wav = wav[np.newaxis, :]

            if not mono:
                wav = np.concatenate((wav, wav), axis=0)

        wav = wav.T
        wav = wav[:sample_rate * 4]

        if np.abs(wav.max()) > 1.0:
            wav = transforms.scale(wav, wav.min(), wav.max(), -1.0, 1.0)

        wav = transforms.scale(wav, wav.min(), wav.max(), -32768.0, 32767.0).T

        return fn, sample_rate, wav.astype(np.float32)

    def load_data(self):
        # read metadata
        meta = pd.read_csv(
            os.path.join(self.root, 'metadata', 'UrbanSound8K.csv'),
            sep=',',
            index_col='slice_file_name'
        )

        for row_idx, (fn, row) in enumerate(meta.iterrows()):
            path = os.path.join(self.root, 'audio', 'fold{}'.format(row['fold']), fn)
            self.data[fn] = path, self.sample_rate, self.mono

        # by default, the official split from the metadata is used
        files_to_load = list()
        # if the random seed is not None, the random split is used
        if self.random_split_seed is not None:
            # given an integer random seed
            skf = skms.StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_split_seed)

            # split the US8K samples into 10 folds
            for fold_idx, (train_ids, test_ids) in enumerate(skf.split(
                    np.zeros(len(meta)), meta['classID'].values.astype(int)
            ), 1):
                # if this is the fold we want to load, add the corresponding files to the list
                if fold_idx == self.fold:
                    ids = train_ids if self.train else test_ids
                    filenames = meta.iloc[ids].index
                    files_to_load.extend(filenames)
                    break
        else:
            # if the random seed is None, use the official split
            for fn, row in meta.iterrows():
                if int(row['fold']) in self.folds_to_load:
                    files_to_load.append(fn)

        self.data = {fn: vals for fn, vals in self.data.items() if fn in files_to_load}
        self.indices = {idx: fn for idx, fn in enumerate(self.data)}

        num_processes = mp.cpu_count()
        warnings.filterwarnings('ignore')
        with mp.Pool(processes=num_processes) as pool:
            chunksize = int(np.ceil(len(meta) / num_processes)) or 1

            tqdm.tqdm.write(f'Loading {self.__class__.__name__} (train={self.train})')

            for fn, sample_rate, wav in pool.starmap(
                func=self._load_worker,
                iterable=[(fn, path, sr, mono) for fn, (path, sr, mono) in self.data.items()],
                chunksize=chunksize
            ):
                self.data[fn] = {
                    'audio': wav,
                    'sample_rate': sample_rate,
                    'target': meta.loc[fn, 'classID']
                }

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        if not (0 <= index < len(self)):
            raise IndexError

        audio: np.ndarray = self.data[self.indices[index]]['audio']
        target: int = self.data[self.indices[index]]['target']

        if self.transform is not None:
            audio = self.transform(audio)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return audio, target

    def __len__(self) -> int:
        return len(self.data)

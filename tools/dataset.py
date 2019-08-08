'''
For accessing dataset and dataset samples only.
'''
import argparse
import glob
import os
import pandas as pd

import numpy as np
import cv2

from math import sqrt
import random

import wget
import zipfile


'''
Labels used in
[1] T. Kawashima et al., "Action recognition from extremely low-resolution  
thermal image sequence," 2017 14th IEEE International Conference on Advanced Video
and Signal Based Surveillance (AVSS), Lecce, 2017, pp. 1-6.

'''
PAPER_LABELS_REGEX = dict([
    (r'walk.*', 0),
    (r'sitdown', 1),
    (r'standup', 2),
    (r'falling.*', 3),
    (r'^(sit|lie|stand)', 4),
])


LABELS_REGEX = dict([
    (r'walk.*', 0),
    (r'sitdown', 1),
    (r'standup', 2),
    (r'falling.*', 3),
    (r'sit', 4),
    (r'lie', 5),
    (r'stand', 6),

])

RAW_LABELS = ('walk', 'walk1', 'walk2',
'sitdown', 'standup',
'falling1', 'falling2',
'sit', 'lie', 'stand')


ACTION_LABELS = ('walk', 
'sitdown', 'standup',
'falling', 
'sit', 'lie', 'stand')

SKIP_FRAMES = 20
DATSAET_URL = "https://github.com/muralab/Low-Resolution-FIR-Action-Dataset/archive/master.zip"
DATASET_FN_ZIP = "Low-Resolution-FIR-Action-Dataset-master.zip"

FILE_COLUMN = 0
ACTOR_COLUMN = 4


ACTORS = np.array([
    'human1', 'human2', 'human3', 'human4', 'human5', 'human6',  
    'human7', 'human8', 'human9'], dtype=object)
SUBJECTS = ACTORS

def download(download_dir: str = "..", dataset_name: str = "dataset"):
    """
    Downloads the dataset (from github) and to dataset_dir and unpacks it to dataset_name

    Args:
    download_dir - where to download the file
    dataset_name - unpacked folder's name
    """
    if os.path.exists(os.path.join(download_dir, dataset_name)):
        print("[ERROR] Folder {} exists. Aborting download of the dataset.".format(os.path.join(dataset_name, download_dir)))
        return
    print("[INFO] Downloading FIR Action Dataset...")
    wget.download(DATSAET_URL, bar=wget.bar_adaptive)
    path, filename = download_dir, dataset_name
    with zipfile.ZipFile(DATASET_FN_ZIP, "r") as zip_ref:
        zip_ref.extractall(path)
    os.remove(DATASET_FN_ZIP)
    dataset_fn = DATASET_FN_ZIP.split(".")[-2]
    os.rename(os.path.join(path, dataset_fn), os.path.join(path, filename))
    print("")
    print("[INFO] Dataset downloaded to %s" % os.path.join(path, filename))
    return


def _load_annotation(dataset_dir: str) -> pd.core.frame.DataFrame:
    pattern = os.path.join(dataset_dir, 'annotation', '*_human.csv')
    generator = glob.iglob(pattern)

    return pd.concat([pd.read_csv(fn, header=None)
                      for fn in generator], ignore_index=True)


def _read_sequence_annotation(sequence_name: str, annotation: pd.core.frame.DataFrame = None) -> list:
    if annotation is None:
        return []
    sequence_annotation_pd = annotation[annotation[0] == sequence_name]
    return sequence_annotation_pd.iloc[:, 1:].values.tolist()


def _list_sequences(dataset_dir: str) -> list:
    pattern = os.path.join(dataset_dir, '*', 'raw', '*.csv')
    generator = glob.iglob(pattern)
    return [sequence for sequence in generator]

def _excludeActor(annotation: pd.core.frame.DataFrame, actor: str) -> pd.core.frame.DataFrame:
    return annotation[annotation[ACTOR_COLUMN] != actor]

def _filterActor(annotation: pd.core.frame.DataFrame, actor: str) -> pd.core.frame.DataFrame:
    return annotation[annotation[ACTOR_COLUMN] == actor]

class Dataset():
    def __init__(self, dataset_dir: str, minmax_normalized: bool = None,  actor:str = None, exclude_actor:str = None, sample: bool = False, samples_k: int = 10):
        if not os.path.exists(dataset_dir):
            raise OSError(2, 'No such file or directory', dataset_dir)
        self.sequences = _list_sequences(dataset_dir)
        if sample:
            self.sequences = random.sample(self.sequences, samples_k)
        self.annotation = _load_annotation(dataset_dir)
        self.directory = dataset_dir
        self.actors = pd.unique(self.annotation[ACTOR_COLUMN])
        if actor:
            self.filterActor(actor)
        if exclude_actor:
            self.excludeActor(exclude_actor)
        self.minmax_normalized = minmax_normalized
        self._dataset_dir = dataset_dir
        self._initial_sequences = self.sequences.copy()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return Sequence(self.sequences[idx], dataset_annotation=self.annotation, minmax_normalized = self.minmax_normalized)

    def excludeActor(self, actor: str):
        self.annotation = _excludeActor(self.annotation, actor)
        self._updateActors()
        self._updateSequences()
        return
    
    def filterActor(self, actor: str):
        self.annotation = _filterActor(self.annotation, actor)
        self._updateActors()
        self._updateSequences()
        return
    
    def reload(self):
        self.sequences = self._initial_sequences
        self.annotation = _load_annotation(self._dataset_dir)
        self._updateActors()
        self._updateSequences()
        return
    
    def _updateActors(self):
        self.actors = pd.unique(self.annotation[ACTOR_COLUMN])

    def _updateSequences(self):
        '''
        Updates sequences by removing any sequence name that is not in annotation
        If needed reload annotation before running updateSequences!
        '''
        annotation_sequences_list = pd.unique(self.annotation[FILE_COLUMN])
        for fn in self.sequences:
            path, sequence_name = os.path.split(fn)
            if sequence_name not in annotation_sequences_list:
                self.sequences.remove(fn)


def _minmax_normalize(array: np.ndarray, norm_min: float, norm_max: float) -> np.ndarray:
    return (array - norm_min)/(norm_max - norm_min)


class Sequence(np.ndarray):
    def __new__(cls, fn: str, dataset_annotation=None, frame_start=None, frame_stop=None, minmax_normalized = None):
        # read dataframe
        dataframe = pd.read_csv(fn, skiprows=[0, 1], header=None)
        # skip time and PTAT columns
        pixels = dataframe.iloc[:, 2:].values
        PTAT = dataframe.iloc[:, 1:2].values
        min = pixels[SKIP_FRAMES:].min()
        max = pixels[SKIP_FRAMES:].max()
        PTAT = PTAT[frame_start:frame_stop]
        if minmax_normalized:
            pixels = _minmax_normalize(pixels, min, max)
            #use numpy's min max even though we have to get 0 and 1...
            #so that any future changes that cause bugs are caught in testing
            min = pixels[SKIP_FRAMES:].min()
            max = pixels[SKIP_FRAMES:].max()
        pixels = pixels[frame_start:frame_stop][:]
        # reshape to [frames, h, w] array
        frames, h, w = pixels.shape[0], (int)(
            sqrt(pixels.shape[1])), (int)(sqrt(pixels.shape[1])) 
        pixels = pixels.reshape([frames, h, w])

        obj = np.asarray(pixels).view(cls)
        # add custom sequence attributes
        obj.filename = fn
        path, sequence_name = os.path.split(fn)
        obj.sequence_name = sequence_name
        obj.dataset_annotation = dataset_annotation
        obj.start = frame_start
        obj.stop = frame_stop
        obj.temp_min = min
        obj.temp_max = max
        obj.PTAT = PTAT
        return obj


    def annotation(self):
        return _read_sequence_annotation(self.sequence_name, self.dataset_annotation)

    def as_uint8(self):
        minmax_normalized = None
        if (self.temp_min == 0) and (self.temp_max == 1):
            minmax_normalized = self.copy()
        else:
            minmax_normalized = _minmax_normalize(self.copy(), self.temp_min, self.temp_max)
        return (255 * minmax_normalized).astype(np.uint8)




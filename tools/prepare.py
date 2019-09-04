import os
from tqdm import tqdm 

import numpy as np

from tools import flow

import shutil

def ensure_structure_exist(dir_list):
    """Makes sure the folder exists on disk.

    Args:
    dir_list: List of path strings.
    """
    for dir_name in dir_list:
        ensure_dir_exists(dir_name)

def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.

    Args:
    dir_name: Path to the folder.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def remove_dir_tree(dir_name):
    if os.path.exists(dir_name):
        print("[INFO] Removing {}...".format(dir_name))
        shutil.rmtree(dir_name)


def sequences_by_actor(dataset, temperature_dir):
    remove_dir_tree(temperature_dir)
    ensure_dir_exists(temperature_dir)
    print("[INFO] Adding samples to {}...".format(temperature_dir))
    for actor in dataset.actors:
        ensure_dir_exists(os.path.join(temperature_dir, actor))
    for sample in tqdm(dataset):
        name = sample.sequence_name.split(".")[0]
        for action in sample.annotation():
            start, stop, label, actor = action
            stop += 1
            fn = label + "_" + name + "_" + str(start) + "_" + str(stop) + ".npy"
            path = os.path.join(temperature_dir, actor, fn)
            unit = sample[start:stop]
            with open(path, 'wb+') as handler:
                np.save(handler, unit, allow_pickle=False)
                handler.flush()
    return

def optical_flow(dataset, flow_dir):
    remove_dir_tree(flow_dir)
    ensure_dir_exists(flow_dir)
    print("[INFO] Calculating optical flow and saving to {}...".format(flow_dir))
    for actor in dataset.actors:
        ensure_dir_exists(os.path.join(flow_dir, actor))
    for sample in tqdm(dataset):
        name = sample.sequence_name.split(".")[0]
        for action in sample.annotation():
            start, stop, label, actor = action
            stop += 1
            fn = label + "_" + name + "_" + str(start) + "_" + str(stop) + ".npy"
            path = os.path.join(flow_dir, actor, fn)
            unit = sample[start:stop]
            flow_array = flow.farneback(unit)
            with open(path, 'wb+') as handler:
                np.save(handler, flow_array, allow_pickle=False)
                handler.flush()
    return

"""
For data augmentation of sequences:
- temperature (frames, H, W)
- flow (frames, H, W, 2) - channels are x, y components
"""
import numpy as np
from numpy.random import randint

def random_rotation(array):
    '''rotate 0, 90, 180 or 270 degrees (1/4 probability for each of the options)

    Parameters
    ----------
    array
        temperature or optical flow sequence, shape (frames, H, W) and (frames, H, W, 2) respectively
    
    Returns
    -------
    np.array
        rotated sequence
    '''
    return np.rot90(array, k=randint(0,4), axes=(1, 2))

def random_flip(array):
    '''flip horizontally or vertically (1/3 probabilty for each) or leave untouched (1/3 probability)

    Parameters
    ----------
    array
        temperature or optical flow sequence, shape (frames, H, W) and (frames, H, W, 2) respectively
    
    Returns
    -------
    np.array
        flipped sequence
    '''
    case = randint(0,3)
    if case == 0:
        return np.flip(array, axis=1)
    elif case == 1:
        return np.flip(array, axis=2)
    else:
        return array

import os
walk =r"human1\walk_20170203_p5_dark1_126_142.npy"
sitdown = r"human1\sitdown_20170203_p6_dark2_128_148.npy"
standup = r"human1\standup_20170203_p8_light2_169_197.npy"
falling = r"human1\falling1_20170203_p14_light1_104_125.npy"
sit = r"human1\sit_20170203_p1_dark3_197_208.npy"
lie = r"human1\lie_20170203_p3_light4_175_199.npy"
stand = r"human1\stand_20170203_p3_light3_186_209.npy"
predictions = []
temperature_dir = "/" + os.path.join("tmps", "cache", "temperature")
flow_dir = "/" + os.path.join("tmps", "cache", "optical_flow")


for action in [walk, sitdown, standup, falling, sit, lie, stand]:
    temp = np.load(os.path.join(temperature_dir, action))
    temp = temp[..., np.newaxis]
    temp = temp.reshape([-1, *temp.shape])
    flow = np.load(os.path.join(flow_dir, action))
    flow = flow.reshape([-1, *flow.shape])



for action in [walk, sitdown, standup, falling, sit, lie, stand]:
    temp = np.load(os.path.join(temperature_dir, action))
    flow = np.load(os.path.join(flow_dir, action))

a = temp
b = flow

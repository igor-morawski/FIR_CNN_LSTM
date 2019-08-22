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

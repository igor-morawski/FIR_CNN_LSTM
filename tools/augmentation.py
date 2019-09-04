"""
For data augmentation of sequences:
- temperature (frames, H, W)
- flow (frames, H, W, 2) - channels are x, y components
"""
import numpy as np
from numpy.random import randint

def random_rotation(array, case = None):
    '''rotate 0, 90, 180 or 270 degrees (1/4 probability for each of the options)
    if case is specified then rotation = case * 90 deg

    Parameters
    ----------
    array
        temperature or optical flow sequence, shape (frames, H, W) and (frames, H, W, 2) respectively
    case 
        integer in range [0,4)
    
    Returns
    -------
    np.array
        rotated sequence
    '''
    if not case: 
        case = randint(0,4)
    return np.rot90(array, k=case, axes=(1, 2))

def random_flip(array, case = None):
    '''flip horizontally or vertically (1/3 probabilty for each) or leave untouched (1/3 probability)
    if case is specified then 0 for vertical flip, 1 for horizonal, else for no augmentation

    Parameters
    ----------
    array
        temperature or optical flow sequence, shape (frames, H, W) and (frames, H, W, 2) respectively
    
    Returns
    -------
    np.array
        flipped sequence
    '''
    if not case: 
        case = randint(0,3)
    if case == 0:
        return np.flip(array, axis=1)
    elif case == 1:
        return np.flip(array, axis=2)
    else:
        return array

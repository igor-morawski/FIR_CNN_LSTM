import numpy as np 
import cv2 

def apply_method(sequence, function, args: list):
    frames, height, width = sequence.shape
    flow = np.empty((frames-1, height, width, 2), dtype=np.float32)
    for idx in range(len(sequence)-1):
        current = sequence[idx]
        consecutive = sequence[idx+1]
        flow_map = function(current, consecutive, *args)
        flow[idx] = flow_map
    return flow

def farneback(sequence: np.ndarray, pyr_scale = 0.5, pyr_levels = 3, winsize = 3, 
    iterations = 3, poly_n = 5, poly_sigma = 1.1, flags = 0):
    args = (None, pyr_scale, pyr_levels, winsize, iterations, poly_n, poly_sigma, flags)
    return apply_method(sequence, cv2.calcOpticalFlowFarneback, args)
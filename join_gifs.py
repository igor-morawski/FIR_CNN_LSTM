'''
stacks corresponding gifs from the passed directories horizontally frame by frame
gifs must be the same lenght

e.g. python join_gifs.py --output=\tmps\visualize\joined \tmps\visualize\temperature \tmps\visualize\optical_flow
'''
import os
import argparse
from glob import glob
import collections

import imageio
import numpy as np

def stack_gifs(*args):
    for arg in args:
        print(len(arg))
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--FPS',
    type=int,
    default=10,
    help='frames per second'
)
    parser.add_argument(
    '--output_dir',
    type=str,
    help='output directory'
)
    FLAGS, unparsed = parser.parse_known_args()

    if len(unparsed) is 0: 
      raise TypeError("no input directories passsed")

    for arg in unparsed:
      if not os.path.exists(arg):
        raise OSError(2, 'No such file or directory', arg)
    
    
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    
    args = unparsed
    directories = [arg for arg in args]
    dir_number = len(directories)

    gif_names = list()
    for arg in args:
        gifs = glob(os.path.join(arg, "*.gif"))
        for gif in gifs:
            _, gif_name = os.path.split(gif)
            gif_names.append(gif_name)

    #filter out gifs that are not in all passed directories
    counter = collections.Counter(gif_names)
    gif_names = [gif for gif in counter if (counter[gif] == dir_number)]
    
    gifs_dict = dict()
    for gif_file in gif_names:
        gifs_list = []
        print(gif_file)
        for directory in directories:  
            gif = imageio.imread(glob(os.path.join(directory, gif_file))[0])
            print(len(gif))
            gifs_list.append(gif)
        gifs_dict[gif_file] = gifs_list
    
    k = 'falling1_20170209_p1_light4_132_164.gif'
    l = 'stand_20170210_p2_light2_810_829.gif'

    for gif_name in gifs_dict.keys():
        gifs = gifs_dict[gif_name]

        control_length = len(gifs[0])
        length_ok = True
        for gif in gifs:
            if len(gif) is not control_length:
                length_ok = False
                break
        if not length_ok:
            continue

    

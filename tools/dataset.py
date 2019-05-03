import argparse
import pandas as pd
import fnmatch
import glob
import os

def csv2pkl(directory = "dataset", destination="cached"):
        pattern = os.path.join(directory, '*', 'raw', '*.csv')
        print("Converting (caching files)...")
        for fn in glob.iglob(pattern):
                csv = pd.read_csv(fn, skiprows=1)
                fn = fn.split("\\")[-1].split(".")[0]
                csv.to_pickle(os.path.join(destination,fn+".pkl"))

def isCached(directory = "dataset", destination="cached", cache=True):
        isCached = True
        pattern = os.path.join(directory, '*', 'raw', '*.csv')
        print("Confirming all csv files are cached...")
        for fn in glob.iglob(pattern):
                fn = fn.split("\\")[-1].split(".")[0]
                fn = os.path.join(destination,fn+".pkl")
                if not os.path.isfile(fn):
                        isCached = False
        print("Files cached: {}".format(isCached))
        if not cache:
                return isCached
        if isCached:
                return isCached
        if not isCached:
                csv2pkl(directory, destination)
                return True

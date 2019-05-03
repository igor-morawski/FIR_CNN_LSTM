import os


def dir_init():
    directories = ["model", "cached"]
    for dir in directories:
        if not os.path.exists(dir):
            os.makedirs(dir)

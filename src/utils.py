# utils.py

import json
import os
import pickle

#################################################
#
# Simple I/O
#
#################################################

def save_file(data, path, verbose=False):
    # Creates intermediate directories if they don't exist
    dir = os.path.dirname(path)
    if (dir != '') and (not os.path.isdir(dir)):
        os.makedirs(dir)

    if verbose:
        print(f"Saving: {path}")

    _, ext = os.path.splitext(path)
    if ext == ".pkl":
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=2)
    elif ext == ".json":
        with open(path, "w") as f:
            json.dump(data, f, indent=4, separators=(",", ": "), sort_keys=True)
            f.write("\n")  # add trailing newline for POSIX compatibility
    elif ext == '.html':
        with open(path, 'w') as f:
            f.write(data)

def load_file(path):
    _, ext = os.path.splitext(path)
    if ext == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif ext == ".json":
        with open(path, "r") as f:
            data = json.load(f)
    return data
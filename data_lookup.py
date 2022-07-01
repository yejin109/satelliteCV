import os
import json
import pandas as pd
import pprint


def show_sample():
    with open(f"{path_labels}/{labels[1513]}") as label:
        result = json.load(label)
    return result


def main():
    sample = show_sample()
    pprint.pprint(sample)


if __name__ == '__main__':
    path_objects = f"data/raw/objects"
    path_labels = f"data/raw/labels"

    files = os.listdir(path_objects)
    labels = os.listdir(path_labels)
    main()
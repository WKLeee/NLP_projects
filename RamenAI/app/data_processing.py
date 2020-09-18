import json
import config as conf
import os

'''

This script loads data from the json file and do the pre-processing work.

'''

def load_file(path):
    data = []
    f = open(path, "r")
    while True:
        s = f.readline()
        if len(s) < 1:
            break
        data.append(json.loads(s))
    f.close()

    return data

if __name__ == "__main__":
    if conf.TEST == 0:
        review_path = conf.REVIEW_DATA_PATH
    else:
        review_path = conf.MINI_DATA_PATH
    data = load_file(review_path)
    print(len(data))
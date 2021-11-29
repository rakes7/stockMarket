import os

import pandas as pd

fileDir = os.path.dirname(os.path.realpath(__file__))


# get a list of crypto starting from a specific folder
def get_crypto_symbols_from_folder(PATH_SOURCE):
    crypto_symbols = []
    for file in os.listdir(PATH_SOURCE):
        crypto = file.replace(".csv", "")
        crypto_symbols.append(crypto)
    return crypto_symbols


# get a list of crypto starting from a specific text file
def get_crypto_symbols_from_text_file():
    crypto_symbols = []
    fileToRead = read_file("../acquisition/crypto_symbols.txt")
    for line in fileToRead:
        crypto_symbols.append(line.replace("\n", ""))
    fileToRead.close()
    return crypto_symbols


def get_dict_symbol_id(PATH):
    df = pd.read_csv(PATH + "symbol_id.csv", sep=",", header=0, index_col=1)
    return df


"""def get_clusters(filename):
    return read_json('crypto_clustering/results/'+filename+".json")

def get_clusters2(path_experiment,filename):
    return read_json('crypto_clustering/'+path_experiment+"/clusters/"+filename+".json")"""


def read_file(path):
    return open(path, "r")


def read_csv(path):
    return pd.read_csv(path)

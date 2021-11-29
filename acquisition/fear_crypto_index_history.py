import requests
import re
import pandas as pd
import os
from utility.folder_creator import folder_creator

URL = "https://api.alternative.me/fng/?limit=0&date_format=cn&format=csv"
REGEX_FOR_INDEX = ".*\"data\": \[(.*)\]"
PATH_FEAR_INDEX = "../acquisition/dataset/fear_index/"


def get_fear_index(start_date, end_date):
    folder_creator(PATH_FEAR_INDEX, 0)
    response = requests.get(URL)
    match = re.search(REGEX_FOR_INDEX, response.text, re.DOTALL)
    if not match:
        raise ValueError('Could not get Crypto Fear & Greed Index')
    else:
        index = match.group(1)
        with open("Output.txt", "w") as text_file:
            print("{}".format(index), file=text_file)
        read_file = pd.read_csv('Output.txt')
        read_file.to_csv(PATH_FEAR_INDEX + 'Fear_Index.csv', index=None)
        os.remove("Output.txt")
        df = pd.read_csv(PATH_FEAR_INDEX + 'Fear_Index.csv')
        df.set_axis(['Date', 'fng_value', 'fng_classification'], axis=1, inplace=True)
        df = df.sort_values('Date', ascending=True)
        df["Date"] = pd.to_datetime(df["Date"], format='%Y-%m-%d')
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        df = df.loc[mask]
        df.to_csv(PATH_FEAR_INDEX + 'Fear_Index.csv', index=None)
    return

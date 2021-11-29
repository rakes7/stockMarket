import os

import pandas as pd

from utility.reader import get_crypto_symbols_from_text_file
from visualization.bar_chart.exploration import missing_values_by_year

COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
PATH_DATA_UNDERSTANDING = "../understanding/output/"


def generate_bar_chart_by_year(PATH_TO_SAVE):
    df = pd.read_csv(PATH_DATA_UNDERSTANDING + "missing_by_year.csv", delimiter=',', header=0)
    df = df.set_index("Symbol")
    missing_values_by_year(df, PATH_TO_SAVE)


def count_missing_values_by_year(PATH_DATASET):
    cryptos = get_crypto_symbols_from_text_file()
    df_out = pd.DataFrame(0, columns=['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021'],
                          index=cryptos)
    for crypto in os.listdir(PATH_DATASET):
        df = pd.read_csv(PATH_DATASET + crypto, delimiter=',', header=0)
        crypto_name = crypto.replace(".csv", "")
        df = df.set_index("Date")
        total_null = df.isnull().sum()[1]
        init_date = df.index[0]
        # get year
        first_year = int(init_date.split("-")[0])
        actual_year = first_year
        while actual_year < 2022:
            # next year
            actual_year += 1
            start_date = str(first_year - 1) + "-12-31"
            end_date = str(actual_year) + "-01-01"
            df1 = df.query('index<@end_date')
            df1 = df1.query('index >@start_date')
            # number of null of this year
            null_year = df1.isnull().sum()[1]
            first_year = actual_year
            df_out.at[crypto_name, str(first_year - 1)] = null_year
    # reset the index, in this way its possibile to have a new column named Symbol
    df_out = df_out.rename_axis('Symbol').reset_index()
    df_out.to_csv(PATH_DATA_UNDERSTANDING + "missing_by_year.csv", ",", index=False)


# generates a file in which, for each cryptocurrency, there is the count of the missing values by column
def count_missing_values(PATH_DATASET):
    crypto = get_crypto_symbols_from_text_file()
    # create a new dataframe
    df = pd.DataFrame(columns=COLUMNS, index=crypto)
    for file in os.listdir(PATH_DATASET):
        df1 = pd.read_csv(PATH_DATASET + file, delimiter=',', header=0)
        df1 = df1.set_index("Date")
        crypto_name = file.replace(".csv", "")
        df.loc[crypto_name] = df1.isnull().sum()  # inserting a series in a dataframe row
        df.to_csv(PATH_DATA_UNDERSTANDING + "count_missing_values.csv", ",")

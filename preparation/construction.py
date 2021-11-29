import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

from utility.folder_creator import folder_creator

# SCALING
"""Normalization is the process of scaling individual samples to have unit norm. 
This process can be useful if you plan to use a quadratic form such 
as the dot-product or any other kernel to quantify the similarity of any pair of samples.
This assumption is the base of the Vector Space Model often used in text classification and clustering contexts.
If you want to cluster based on similar shape in the cluster rather then similar variance (standardization)"""


def min_max_scaling(input_path, output_path):
    folder_creator(output_path, 1)
    excluded_features = ['Date']
    for crypto in os.listdir(input_path):
        df = pd.read_csv(input_path + crypto, delimiter=',', header=0)
        scaler = MinMaxScaler()
        for col in df.columns:
            if col not in excluded_features:
                normalized = scaler.fit_transform(df[col].values.reshape(-1, 1))
                df[col] = pd.Series(normalized.reshape(-1))
        # todo we have to round 8 since the neural network takes floating numbers with this limit (df.round(8))
        df.to_csv(output_path + crypto, sep=",", index=False)


# SCALING
"""Normalization is the process of scaling individual samples to have unit norm. 
This process can be useful if you plan to use a quadratic form such 
as the dot-product or any other kernel to quantify the similarity of any pair of samples.
This assumption is the base of the Vector Space Model often used in text classification and clustering contexts.
If you want to cluster based on similar shape in the cluster rather then similar variance (standardization)"""


def min_max_one_minusone_scaling(input_path, output_path):
    folder_creator(output_path, 1)
    excluded_features = ['Date']
    for crypto in os.listdir(input_path):
        df = pd.read_csv(input_path + crypto, delimiter=',', header=0)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        for col in df.columns:
            if col not in excluded_features:
                normalized = scaler.fit_transform(df[col].values.reshape(-1, 1))
                df[col] = pd.Series(normalized.reshape(-1))
        # todo we have to round 8 since the neural network takes floating numbers with this limit (df.round(8))
        df.to_csv(output_path + crypto, sep=",", index=False)


# SCALING

def robust_scaling(input_path, output_path):
    folder_creator(output_path, 1)
    excluded_features = ['Date']
    for crypto in os.listdir(input_path):
        df = pd.read_csv(input_path + crypto, delimiter=',', header=0)
        scaler = RobustScaler()
        for col in df.columns:
            if col not in excluded_features:
                normalized = scaler.fit_transform(df[col].values.reshape(-1, 1))
                df[col] = pd.Series(normalized.reshape(-1))
        df.to_csv(output_path + crypto, sep=",", index=False)


def max_abs_scaling(input_path, output_path):
    folder_creator(output_path, 1)
    excluded_features = ['Date', 'trend']
    for crypto in os.listdir(input_path):
        splitted = crypto.split("_")
        crypto_name = splitted[0]
        folder_creator(os.path.join(output_path, crypto_name), 0)
        df = pd.read_csv(os.path.join(input_path, crypto), delimiter=',', header=0)
        day_to_predict = df.loc[len(df.Date) - 1]
        df = df[:-1]  # remove the date to predict
        scaler = MaxAbsScaler()
        for col in df.columns:
            if col not in excluded_features:
                normalized = scaler.fit_transform(df[col].values.reshape(-1, 1))
                df[col] = pd.Series(normalized.reshape(-1))
        df = df.append(day_to_predict, ignore_index=True)
        df.to_csv(os.path.join(output_path, crypto_name, crypto), sep=",", index=False)


def standardization(input_path, output_path):
    folder_creator(output_path, 1)
    excluded_features = ['Date']
    for crypto in os.listdir(input_path):
        df = pd.read_csv(input_path + crypto, delimiter=',', header=0)
        scaler = StandardScaler()
        for col in df.columns:
            if col not in excluded_features:
                normalized = scaler.fit_transform(df[col].values.reshape(-1, 1))
                df[col] = pd.Series(normalized.reshape(-1))
        df.to_csv(output_path + crypto, sep=",", index=False)


# creates the horizontal dataset
def create_horizontal_dataset(data_path, output_path, test_set):
    cryptocurrencies_with_date_to_pred = os.listdir(data_path)
    cryptos_in_the_cluster = []
    already_created = False
    folder_creator(output_path + "horizontal_datasets" + "/", 0)
    print("Creating horizontal version")
    for date_to_predict in test_set:
        dictionary_m = {}
        dataframes = []
        # take just the date column one time
        for dataset_name in cryptocurrencies_with_date_to_pred:
            splitted = dataset_name.split("_")
            date_to_predict_crypto = str(splitted[1]).replace(".csv", "")
            if date_to_predict == date_to_predict_crypto:
                df_date = pd.read_csv(os.path.join(data_path, dataset_name))
                dataframes.append(df_date['Date'])
                break

        # creates Close_1,Open_1 ecc for each dataframe
        i = 1
        for dataset_name in cryptocurrencies_with_date_to_pred:
            splitted = dataset_name.split("_")
            crypto_name = splitted[0]
            date_to_predict_crypto = str(splitted[1]).replace(".csv", "")
            if date_to_predict == date_to_predict_crypto:
                df = pd.read_csv(os.path.join(data_path, dataset_name), header=0)
                if already_created == False:
                    cryptos_in_the_cluster.append(crypto_name)
                df = df.drop('Date', axis=1)
                df['symbol'] = crypto_name
                df = df.add_suffix('_' + str(i))
                i += 1
                # dictionary_m[crypto_name]=crypto_name
                dictionary_m[crypto_name + 'dataframe'] = df
        for crypt in cryptos_in_the_cluster:
            dataframes.append(dictionary_m.get(crypt + 'dataframe'))
        already_created = True
        # concat horizontally all the dataframes
        horizontal = pd.concat(dataframes, axis=1)
        # serialization
        horizontal.to_csv(output_path + "horizontal_datasets/horizontal_" + date_to_predict + ".csv", sep=",",
                          index=False)
        del horizontal
        del dataframes
        del dictionary_m
        print("Horizontal version created for the date: " + str(date_to_predict))
    return list(cryptos_in_the_cluster)


# [close(i+1)-close(i)/close(i)*100]
def add_trend_feature(input_path, output_path, percent):
    for crypto in os.listdir(input_path):
        df = pd.read_csv(os.path.join(input_path, crypto), sep=",", header=0)
        df['pct_change'] = df['Close'].pct_change()
        df['pct_change'] = df['pct_change'].apply(lambda x: x * 100)
        # 0 is stable
        # 1 is down
        # 2 is up
        df['trend'] = 0
        df.loc[df['pct_change'] < -percent, 'trend'] = -1  # down
        df.loc[df['pct_change'] > percent, 'trend'] = 1  # up
        # print(df[['pct_change','trend']])
        df.to_csv(output_path + crypto, sep=",", index=False)


def change_relative_variance(input_path, output_path):
    folder_creator(output_path, 1)
    for crypto in os.listdir(input_path):
        splitted = crypto.split("_")
        crypto_name = splitted[0]
        folder_creator(os.path.join(output_path, crypto_name), 0)
        df = pd.read_csv(os.path.join(input_path, crypto), sep=",", header=0)
        df[df.columns.drop(['Date', 'trend'])] = ((df[df.columns.drop(['Date', 'trend'])] - df[
            df.columns.drop(['Date', 'trend'])].shift(1)) / df[df.columns.drop(['Date', 'trend'])].shift(1)) * 100
        df = df.iloc[1:, :]
        df.to_csv(os.path.join(output_path, crypto_name, crypto), sep=",", index=False)


"""def add_trend_feature(input_path,output_path,percent):
    for crypto in os.listdir(input_path):
        df= pd.read_csv(os.path.join(input_path,crypto),sep=",",header=0)

        df['trend']=0
        for day in df.Date.values:
            if day!=df.Date.values[0]:
                day_before = (pd.to_datetime(day, format="%Y-%m-%d") - timedelta(days=1)).strftime('%Y-%m-%d')

                row_day_before = df[df['Date'] == day_before]
                row_day_before = row_day_before.set_index('Date')

                row_current_day = df[df['Date'] == day]
                row_current_day  = row_current_day .set_index('Date')

                delta_percent=np.multiply(
                    np.divide(np.subtract(row_current_day.loc[day,'Close'],
                                          row_day_before.loc[day_before,'Close']),
                              row_day_before.loc[day_before,'Close']),100)

                print(delta_percent)
                df = df.set_index("Date")
                if delta_percent>percent:
                    #Up:2
                    df.at[day,'trend']=2
                elif delta_percent<percent:
                    #down:1
                    df.at[day, 'trend']=1
                else:
                    pass
                df=df.reset_index()
        df.to_csv(output_path + crypto, sep=",", index=False)
        break"""

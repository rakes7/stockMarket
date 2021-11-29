import os
import shutil

import pandas as pd
from sklearn.preprocessing import RobustScaler

from utility.dataset_utils import cut_dataset_by_range
from utility.folder_creator import folder_creator

PATH_DATASET = "../acquisition/dataset/original/"
PATH_PREPROCESSED_FOLDER = "../preparation/preprocessed_dataset/"
PATH_SENTIMENT = "../acquisition/dataset/fear_index/"
PATH_UNCOMPLETE_FOLDER = "../preparation/preprocessed_dataset/selected/uncomplete/"
PATH_COMPLETE_FOLDER = "../preparation/preprocessed_dataset/selected/complete/"
PATH_CLEANED_FOLDER = "../preparation/preprocessed_dataset/cleaned/"
PATH_NO_MISSING_VALUES_SENTIMENT_FOLDER = "../preparation/preprocessed_dataset/cleaned/no_missing_values/fear_index/"
PATH_NO_MISSING_VALUES_CRYPTO_FOLDER = "../preparation/preprocessed_dataset/cleaned/no_missing_values/crypto/"


def remove_uncomplete_rows_by_range(crypto_symbol, start_date, end_date):
    folder_creator(PATH_CLEANED_FOLDER, 0)
    folder_creator(PATH_CLEANED_FOLDER + "partial", 0)
    df = cut_dataset_by_range(PATH_UNCOMPLETE_FOLDER, crypto_symbol, start_date, end_date)
    df.to_csv(PATH_CLEANED_FOLDER + "partial/" + crypto_symbol + ".csv", sep=",", index=False)


def input_missing_values():
    folder_creator(PATH_CLEANED_FOLDER + "final", 1)
    # already_treated=['LKK.csv','FAIR.csv']
    """for crypto_symbol in os.listdir(PATH_CLEANED_FOLDER+"partial"):
        df = pd.read_csv(PATH_CLEANED_FOLDER+"partial/"+crypto_symbol, delimiter=',', header=0)
        already_treated.append(crypto_symbol)
        df=interpolate_with_time(df)
        df.to_csv(PATH_CLEANED_FOLDER + "final/" + crypto_symbol, sep=",", index=False)"""

    for crypto_symbol in os.listdir(PATH_UNCOMPLETE_FOLDER):
        df = pd.read_csv(PATH_UNCOMPLETE_FOLDER + crypto_symbol, delimiter=',', header=0)
        # if crypto_symbol not in already_treated:
        df = interpolate_with_time(df)
        df.to_csv(PATH_CLEANED_FOLDER + "final/" + crypto_symbol, sep=",", index=False)

    # merge with complete dataset
    for crypto_symbol in os.listdir(PATH_COMPLETE_FOLDER):
        shutil.copy(PATH_COMPLETE_FOLDER + crypto_symbol, PATH_CLEANED_FOLDER + "final/" + crypto_symbol)


# todo spiegare come mai hai scelto questo metodo di interpolazione... ce ne sono tanti a disposizione
def interpolate_with_time(df):
    # Converting the column to DateTime format
    df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df = df.set_index('Date')
    # interpolate with time
    df = df.interpolate(method='time')
    df = df.reset_index()
    return df


# Univariata, ma va bene lo stesso.
def remove_outliers_one():
    folder_creator(PATH_CLEANED_FOLDER + "final/", 1)
    for crypto in os.listdir(PATH_COMPLETE_FOLDER):
        df = pd.read_csv(PATH_COMPLETE_FOLDER + crypto, sep=",", header=0)
        # df=cut_dataset_by_range(PATH_COMPLETE_FOLDER,crypto.replace(".csv",""),'2017-06-27','2019-12-31')
        # df_orig = cut_dataset_by_range(PATH_COMPLETE_FOLDER, crypto.replace(".csv", ""), '2017-08-22', '2019-12-31')
        df.to_csv(PATH_CLEANED_FOLDER + "final/" + crypto, sep=",", index=False)
        # df.to_csv(PATH_CLEANED_FOLDER + "final/" + crypto, sep=",", index=False)

        """low=0.20
        high=0.95
        res=df.Close.quantile([low,high])
        print(res)
        true_index=(res.loc[low] <= df.Close.values) & (df.Close.values <= res.loc[high])
        false_index=~true_index"""
        # df.Close=df.Close[true_index]
        i = 0
        """for index in false_index:
            if index==True:
                if i!=0 and res.loc[low]<=df_orig.Close[i-1] and df_orig.Close[i-1]<=res.loc[high]:
                    df.Close[i]=df_orig.Close[i-1]
                elif i!=0 and df_orig.Close[i-1]<=res.loc[low]:
                    df.Close[i]=res.loc[low]
                elif i!=0 and df_orig.Close[i-1]>=res.loc[high]:
                    df.Close[i] = res.loc[high]
                else:
                    df.Close[i] =res.loc[low]
            i+=1"""
        # df[true_index]=df.Close[true_index]
        """print("Open")
        low = 0.02
        high = 0.85
        res = df.Open.quantile([low, high])
        print(res)
        true_index = (res.loc[low] <= df.Open.values) & (df.Open.values <= res.loc[high])
        false_index = ~true_index
        i = 0
        for index in false_index:
            if index == True:
                if i != 0 and res.loc[low] <= df_orig.Open[i - 1] and df_orig.Open[i - 1] <= res.loc[high]:
                    df.Open[i] = df_orig.Open[i - 1]
                elif i != 0 and df_orig.Open[i - 1] <= res.loc[low]:
                    df.Open[i] = res.loc[low]
                elif i != 0 and df_orig.Open[i - 1] >= res.loc[high]:
                    df.Open[i] = res.loc[high]
                else:
                    df.Open[i] = res.loc[low]
            i += 1"""

        # df[true_index].to_csv(PATH_CLEANED_FOLDER+"final/"+crypto,sep=",",index=False)


from sklearn.cluster import DBSCAN


# usa complete folder (open,high,low and close)
def remove_outliers_dbscan():
    folder_creator(PATH_CLEANED_FOLDER + "/final", 1)
    excluded_features = ['Date']
    for crypto in os.listdir(PATH_COMPLETE_FOLDER):
        # uses all features
        df = pd.read_csv(PATH_COMPLETE_FOLDER + crypto, sep=",", header=0)

        scaler = RobustScaler()
        for col in df.columns:
            if col not in excluded_features:
                normalized = scaler.fit_transform(df[col].values.reshape(-1, 1))
                df[col] = pd.Series(normalized.reshape(-1))

        model = DBSCAN(eps=0.1, min_samples=18).fit(df.drop('Date', axis=1))

        print(len(df[model.labels_ == -1].values))
        labels = model.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print("numb of clusters: " + str(n_clusters_))
        print("numb of outliers: " + str(n_noise_))

        # outliers
        # print(df[model.labels_==-1])

        # saving the not normalized one
        df = pd.read_csv(PATH_COMPLETE_FOLDER + crypto, sep=",", header=0)
        """df.Close[model.labels_ == -1]=np.median(df.Close[model.labels_ != -1])
        df.Open[model.labels_ == -1] = np.median(df.Open[model.labels_ != -1])
        df.High[model.labels_ == -1] = np.median(df.High[model.labels_ != -1])
        df.Low[model.labels_ == -1] = np.median(df.Low[model.labels_ != -1])"""
        # print(df[model.labels_==-1].Close)
        # print(model.labels_)
        df[model.labels_ != -1].to_csv(PATH_CLEANED_FOLDER + "final/" + crypto, sep=",", index=False)


def fill_missing_values_crypto():
    folder_creator(PATH_NO_MISSING_VALUES_CRYPTO_FOLDER, 1)
    for crypto in os.listdir(PATH_DATASET):
        df = pd.read_csv(PATH_DATASET + crypto, delimiter=',', header=0)
        df = df.fillna(method="bfill", axis=0, inplace=False)
        df.to_csv(PATH_NO_MISSING_VALUES_CRYPTO_FOLDER + crypto, sep=",", index=False)


def fill_missing_values_fear_index():
    folder_creator(PATH_NO_MISSING_VALUES_SENTIMENT_FOLDER, 1)
    df = pd.read_csv(PATH_SENTIMENT + "Fear_Index.csv", delimiter=",", header=0)
    row = df.loc[df['Date'] == '2018-04-13']
    row['Date'] = '2018-04-14'
    df = df.append(row, ignore_index=True)
    row['Date'] = '2018-04-15'
    df = df.append(row, ignore_index=True)
    row['Date'] = '2018-04-16'
    df = df.append(row, ignore_index=True)
    df = df.sort_values('Date', ascending=True)
    df.to_csv(PATH_NO_MISSING_VALUES_SENTIMENT_FOLDER + "Fear_Index.csv", sep=",", index=False)

import os
from datetime import timedelta

import pandas as pd
import pandas_ta as panda

from utility.dataset_utils import cut_dataset_by_range
from utility.folder_creator import folder_creator

PATH_TRANSFORMED_FOLDER = "../preparation/preprocessed_dataset/transformed/"
TARGET_FEATURE = "Close"
# Loockbacks extracted from cryptocompare, in the chart section filters
LOOKBACK_RSI = [14, 21, 100, 200]
LOOKBACK_EMA = [5, 12, 26, 50, 100, 200]
LOOKBACK_SMA = [5, 13, 20, 30, 50, 100, 200]
LOOKBACK_TEST = [14, 21, 5, 12, 26, 13, 30, 20, 50, 100, 200]


def integrate_with_indicators(input_path, output_path, start_date, test_set):
    folder_creator(output_path, 1)
    for crypto in os.listdir(input_path):
        for date_to_predict in test_set:
            # end_date=pd.to_datetime(date_to_predict) - timedelta(days=1)
            end_date = date_to_predict
            df = cut_dataset_by_range(input_path, crypto.replace(".csv", ""), start_date, end_date,
                                      features_to_use=None)
            df["Date"] = pd.to_datetime(df["Date"])
            day_to_predict = df.loc[len(df.Date) - 1]
            df = df[:-1]  # remove the day to predict
            # df = df.sort_values('Date', ascending=True)
            data_series_of_target_feature = df[TARGET_FEATURE]
            for lookback_value in LOOKBACK_TEST:
                df['VWAP'] = panda.vwap(df['High'], df['Low'], df['Close'], df['Volume'], lookback_value=lookback_value)
            for lookback_value in LOOKBACK_TEST:
                df[str('SMA_' + str(lookback_value))] = get_SMA(data_series_of_target_feature, lookback_value)
            for lookback_value in LOOKBACK_TEST:
                df[str('EMA_' + str(lookback_value))] = get_EMA(data_series_of_target_feature, lookback_value)
            for lookback_value in LOOKBACK_TEST:
                df[str('RSI_' + str(lookback_value))] = get_RSI(data_series_of_target_feature, lookback_value)

            df_macd = get_MACD(data_series_of_target_feature)
            df['MACD_12_26_9'] = df_macd['MACD_12_26_9']
            df['MACDH_12_26_9'] = df_macd['MACDH_12_26_9']
            df['MACDS_12_26_9'] = df_macd['MACDS_12_26_9']

            df_bbs = panda.bbands(data_series_of_target_feature)
            df['BBL_20'] = df_bbs['BBL_20']
            df['BBM_20'] = df_bbs['BBM_20']
            df['BBU_20'] = df_bbs['BBU_20']

            df['MOM'] = panda.mom(data_series_of_target_feature)

            df_stoch = panda.stoch(df['High'], df['Low'], df['Close'])
            df['STOCHF_14'] = df_stoch['STOCHF_14']
            df['STOCHF_3'] = df_stoch['STOCHF_3']
            df['STOCH_5'] = df_stoch['STOCH_5']
            df['STOCH_3'] = df_stoch['STOCH_3']
            df['CMO'] = panda.cmo(data_series_of_target_feature)
            df['DPO'] = panda.dpo(data_series_of_target_feature)
            df['UO'] = panda.uo(df['High'], df['Low'], df['Close'])

            df['lag_1'] = df['Close'].shift(1)
            """df['lag_7'] = df['Close'].shift(7)
            df = df.iloc[7:]"""

            df = df.append(day_to_predict, ignore_index=True)
            df.fillna(value=0, inplace=True)
            df.to_csv(os.path.join(output_path, crypto.replace(".csv", "") +
                                   str("_" + date_to_predict) + ".csv"), sep=",", index=False)


def integrate_with_indicators_forecasting(input_path, output_path, start_date, test_set, end_date_for_clustering):
    folder_creator(PATH_INTEGRATED_FOLDER, 1)
    for crypto in os.listdir(input_path):
        for date_to_predict in test_set:
            # end_date=pd.to_datetime(date_to_predict) - timedelta(days=1)
            end_date = date_to_predict
            df = cut_dataset_by_range(input_path, crypto.replace(".csv", ""), start_date, end_date,
                                      features_to_use=None)
            df["Date"] = pd.to_datetime(df["Date"])
            day_to_predict = df.loc[len(df.Date) - 1]
            df = df[:-1]  # remove the day to predict
            # df = df.sort_values('Date', ascending=True)
            data_series_of_target_feature = df[TARGET_FEATURE]
            for lookback_value in LOOKBACK_TEST:
                df['VWAP'] = panda.vwap(df['High'], df['Low'], df['Close'], df['Volume'], lookback_value=lookback_value)
            for lookback_value in LOOKBACK_TEST:
                df[str('SMA_' + str(lookback_value))] = get_SMA(data_series_of_target_feature, lookback_value)
            for lookback_value in LOOKBACK_TEST:
                df[str('EMA_' + str(lookback_value))] = get_EMA(data_series_of_target_feature, lookback_value)
            for lookback_value in LOOKBACK_TEST:
                df[str('RSI_' + str(lookback_value))] = get_RSI(data_series_of_target_feature, lookback_value)

            df_macd = get_MACD(data_series_of_target_feature)
            print(df_macd.columns)
            df['MACD_12_26_9'] = df_macd['MACD_12_26_9']
            df['MACDH_12_26_9'] = df_macd['MACDH_12_26_9']
            df['MACDS_12_26_9'] = df_macd['MACDS_12_26_9']

            df_bbs = panda.bbands(data_series_of_target_feature)
            df['BBL_20'] = df_bbs['BBL_20']
            df['BBM_20'] = df_bbs['BBM_20']
            df['BBU_20'] = df_bbs['BBU_20']

            df['MOM'] = panda.mom(data_series_of_target_feature)

            df_stoch = panda.stoch(df['High'], df['Low'], df['Close'])
            df['STOCHF_14'] = df_stoch['STOCHF_14']
            df['STOCHF_3'] = df_stoch['STOCHF_3']
            df['STOCH_5'] = df_stoch['STOCH_5']
            df['STOCH_3'] = df_stoch['STOCH_3']
            df['CMO'] = panda.cmo(data_series_of_target_feature)
            df['DPO'] = panda.dpo(data_series_of_target_feature)
            df['UO'] = panda.uo(df['High'], df['Low'], df['Close'])

            df['lag_1'] = df['Close'].shift(1)
            """df['lag_7'] = df['Close'].shift(7)
            df = df.iloc[7:]"""

            df = df.append(day_to_predict, ignore_index=True)
            df.fillna(value=0, inplace=True)
            df.to_csv(os.path.join(output_path, crypto.replace(".csv", "") +
                                   str("_" + date_to_predict) + ".csv"), sep=",", index=False)


def get_MACD(data_series_of_target_feature):
    return panda.macd(data_series_of_target_feature)


def get_RSI(data_series_of_target_feature, lookback_value):
    return panda.rsi(data_series_of_target_feature, length=lookback_value)


def get_SMA(data_series_of_target_feature, lookback_value):
    return panda.sma(data_series_of_target_feature, length=lookback_value)


def get_EMA(data_series_of_target_feature, lookback_value):
    return panda.ema(data_series_of_target_feature, length=lookback_value)


def integrate_with_lag(input_path):
    folder_creator(PATH_INTEGRATED_FOLDER, 1)
    for crypto in os.listdir(input_path):
        df = pd.read_csv(input_path + crypto, sep=',', header=0)
        df["Date"] = pd.to_datetime(df["Date"])
        df['lag_1'] = df['Close'].shift(1)
        df['lag_2'] = df['Close'].shift(2)
        df['lag_3'] = df['Close'].shift(3)
        df['lag_7'] = df['Close'].shift(7)
        df = df.iloc[7:]
        df.to_csv(PATH_INTEGRATED_FOLDER + "/" + crypto, sep=",", index=False)

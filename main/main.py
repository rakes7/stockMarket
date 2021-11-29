import os
from datetime import datetime
from decimal import Decimal
from itertools import product
from math import sqrt
import pandas as pd
from acquisition.fear_crypto_index_history import get_fear_index
from modelling.techniques.baseline.simple_prediction.simple_prediction import simple_prediction
from modelling.techniques.baseline.vector_autoregression.vector_autoregression import vector_autoregression
from modelling.techniques.clustering.clustering import clustering
from modelling.techniques.clustering.visualization import describe_new
from modelling.techniques.forecasting.multi_target import multi_target
from modelling.techniques.forecasting.single_target import single_target
from modelling.techniques.forecasting.testing.test_set import generate_testset, get_testset, \
    generate_testset_baseline
from preparation.construction import create_horizontal_dataset
from preparation.preprocessing import preprocessing
from acquisition.yahoo_finance_history import get_most_important_cryptos
from understanding.exploration import describe, missing_values
from utility.clustering_utils import merge_predictions
from utility.dataset_utils import cut_dataset_by_range
from utility.folder_creator import folder_creator
from visualization.bar_chart.forecasting import comparison_macro_avg_recall_single_vs_baseline, \
    overall_comparison_macro_avg_recall_simple_vs_baseline, \
    overall_macro_avg_recall_single, report_multi_target_crypto_oriented, report_multi_target_k_oriented, \
    report_multi_target_overall, report_crypto, comparison_macro_avg_recall_single_sentiment_vs_single_no_sentiment, \
    overall_comparison_macro_avg_recall_single_sentiment_vs_single_no_sentiment, \
    report_multi_target_overall_multi_sentiment_vs_no_sentiment, \
    report_multi_target_crypto_oriented_sentiment_vs_no_sentiment, report_multi_target_k_oriented_vs_single_target, \
    report_multi_target_overall_vs_single_target
from visualization.line_chart import generate_line_chart
import numpy as np


def main():
    # DATA UNDERSTANDING
    cryptos = ["BTC", "BTS", "DASH","DGB","DOGE", "ETH", "IOC", "LTC", "MAID", "MONA", "NAV", "SYS", "VTC", "XCP",
               "XLM", "XMR", "XRP"]
    #data_understanding(cryptos)

    # TESTING SET
    TEST_SET = testing_set()
    start_date = "2018-02-01"
    end_date_for_clustering = "2020-05-31"
    variation = [1, 2]
    percentual_of_variation = 1
    # DATA PREPARATION
    #preprocessing(TEST_SET, start_date, percentual_of_variation, end_date_for_clustering)

    # CLUSTERING
    distance_measure = "pearson"
    features_to_use = ['Close']
    type_clustering = "min_max_normalized"
    # Description after
    type = "max_abs_normalized"
    # clustering
    #type_clustering = "relative_variance"
    #type = "relative_variance"
    '''
    clustering(distance_measure, type_for_clustering=type_clustering, type_for_prediction=type,
               features_to_use=features_to_use)
  
    describe(PATH_DATASET="../preparation/preprocessed_dataset/constructed/" + type + "/",
             output_path="../preparation/preprocessed_dataset/",
             name_folder_res=type)
    '''

    #for percentual_of_variation in variation:
        # MODELLING
        # NO SENTIMENT
    features_to_use = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'VWAP',
                           'SMA_14', 'SMA_21', 'SMA_5', 'SMA_12', 'SMA_26', 'SMA_13', 'SMA_30',
                           'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200', 'EMA_14', 'EMA_21', 'EMA_5',
                           'EMA_12', 'EMA_26', 'EMA_13', 'EMA_30', 'EMA_20', 'EMA_50', 'EMA_100',
                           'EMA_200', 'RSI_14', 'RSI_21', 'RSI_5', 'RSI_12', 'RSI_26', 'RSI_13',
                           'RSI_30', 'RSI_20', 'RSI_50', 'RSI_100', 'RSI_200', 'MACD_12_26_9',
                           'MACDH_12_26_9', 'MACDS_12_26_9', 'BBL_20', 'BBM_20', 'BBU_20', 'MOM', 'CMO', 'UO',
                           'trend']
        #features_to_use = ['Date','Close', 'trend']
        # General parameters
    temporal_sequences = [30, 45, 60]
    list_number_neurons = [128, 256]
    learning_rate = 0.001
    DROPOUT = 0.45
    EPOCHS = 500
    PATIENCE = 4
    BATCH_SIZE = None

    single_target_main(TEST_SET, type, features_to_use,
                           temporal_sequences, list_number_neurons, learning_rate, DROPOUT,
                           EPOCHS, PATIENCE, BATCH_SIZE, percentual_of_variation)


    # WITH SENTIMENT
    features_to_use = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'VWAP',
                           'SMA_14', 'SMA_21', 'SMA_5', 'SMA_12', 'SMA_26', 'SMA_13', 'SMA_30',
                           'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200', 'EMA_14', 'EMA_21', 'EMA_5',
                           'EMA_12', 'EMA_26', 'EMA_13', 'EMA_30', 'EMA_20', 'EMA_50', 'EMA_100',
                           'EMA_200', 'RSI_14', 'RSI_21', 'RSI_5', 'RSI_12', 'RSI_26', 'RSI_13',
                           'RSI_30', 'RSI_20', 'RSI_50', 'RSI_100', 'RSI_200', 'MACD_12_26_9',
                           'MACDH_12_26_9', 'MACDS_12_26_9', 'BBL_20', 'BBM_20', 'BBU_20', 'MOM', 'CMO', 'UO',
                           'sentiment', 'trend']
    """single_target_main(TEST_SET, type, features_to_use,
                           temporal_sequences, list_number_neurons, learning_rate, DROPOUT,
                           EPOCHS, PATIENCE, BATCH_SIZE, percentual_of_variation, sentiment=1)"""
    # report_single_target_sentiment_vs_no_sentiment(percentual_of_variation)

    # MULTITARGET
    temporal_sequences = [30, 45, 60]
    list_number_neurons = [128, 256]
    learning_rate = 0.001
    DROPOUT = 0.45
    EPOCHS = 500
    PATIENCE = 4
    BATCH_SIZE = None
    percentual_of_variation = 1
    clusters = ["cluster_9"]

    # NO SENTIMENT
    features_to_use = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'VWAP',
                       'SMA_14', 'SMA_21', 'SMA_5', 'SMA_12', 'SMA_26', 'SMA_13', 'SMA_30',
                       'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200', 'EMA_14', 'EMA_21', 'EMA_5',
                       'EMA_12', 'EMA_26', 'EMA_13', 'EMA_30', 'EMA_20', 'EMA_50', 'EMA_100',
                       'EMA_200', 'RSI_14', 'RSI_21', 'RSI_5', 'RSI_12', 'RSI_26', 'RSI_13',
                       'RSI_30', 'RSI_20', 'RSI_50', 'RSI_100', 'RSI_200', 'MACD_12_26_9',
                       'MACDH_12_26_9', 'MACDS_12_26_9', 'BBL_20', 'BBM_20', 'BBU_20', 'MOM', 'CMO', 'UO',
                       'trend']
    """multi_target_main(TEST_SET, features_to_use, temporal_sequences, list_number_neurons, learning_rate, DROPOUT,
                      EPOCHS, PATIENCE, clusters, percentual_of_variation, BATCH_SIZE)"""

    # WITH SENTIMENT
    features_to_use = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'VWAP',
                       'SMA_14', 'SMA_21', 'SMA_5', 'SMA_12', 'SMA_26', 'SMA_13', 'SMA_30',
                       'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200', 'EMA_14', 'EMA_21', 'EMA_5',
                       'EMA_12', 'EMA_26', 'EMA_13', 'EMA_30', 'EMA_20', 'EMA_50', 'EMA_100',
                       'EMA_200', 'RSI_14', 'RSI_21', 'RSI_5', 'RSI_12', 'RSI_26', 'RSI_13',
                       'RSI_30', 'RSI_20', 'RSI_50', 'RSI_100', 'RSI_200', 'MACD_12_26_9',
                       'MACDH_12_26_9', 'MACDS_12_26_9', 'BBL_20', 'BBM_20', 'BBU_20', 'MOM', 'CMO', 'UO',
                       'sentiment', 'trend']
    """multi_target_main(TEST_SET, features_to_use, temporal_sequences, list_number_neurons, learning_rate, DROPOUT,
                      EPOCHS, PATIENCE, clusters, percentual_of_variation, BATCH_SIZE, sentiment=1)"""

    """report_multi_target_sentiment_vs_no_sentiment(percentual_of_variation)"""


def multi_target_main(TEST_SET, features_to_use, temporal_sequences, list_number_neurons, learning_rate,
                      DROPOUT, EPOCHS, PATIENCE, clusters, percent, BATCH_SIZE=None, sentiment=0):
    types = ["k_sqrtN", "k_sqrtNBy2", "k_sqrtNBy4", "k_sqrtNDiv2", "k_sqrtNDiv4"]
    DATA_PATH = "../modelling/techniques/clustering/output/clusters/"
    for cluster_n in clusters:
        if sentiment:
            EXPERIMENT_PATH = "../modelling/techniques/forecasting/outputs_multi_" + str(
                percent) + "nuovo_senza_pesi/outputs_multi_sentiment_" + str(
                percent) + "/multi_target/"
        else:
            EXPERIMENT_PATH = "../modelling/techniques/forecasting/outputs_multi_" + str(
                percent) + "nuovo_senza_pesi/outputs_multi_no_sentiment_" + str(
                percent) + "/multi_target/"
        folder_creator(EXPERIMENT_PATH + "clusters" + "/" + cluster_n + "/", 0)
        # generate horizontal dataset
        cryptos_in_cluster = create_horizontal_dataset(DATA_PATH + cluster_n + "/",
                                                       EXPERIMENT_PATH + "clusters" + "/" + cluster_n + "/",
                                                       TEST_SET)

        features_to_use_multi = ['Date']
        for i in range(len(cryptos_in_cluster)):
            for feature in features_to_use:
                features_to_use_multi.append(feature + "_" + str(i + 1))

        multi_target(EXPERIMENT_PATH=EXPERIMENT_PATH + "clusters" + "/" + cluster_n + "/",
                     DATA_PATH=EXPERIMENT_PATH + "clusters" + "/" + cluster_n + "/" + "horizontal_datasets/",
                     TENSOR_DATA_PATH=EXPERIMENT_PATH + "clusters" + "/" + cluster_n + "/tensor_data/",
                     window_sequences=temporal_sequences,
                     list_num_neurons=list_number_neurons, learning_rate=learning_rate,
                     cryptos=cryptos_in_cluster,
                     features_to_use=features_to_use_multi,
                     DROPOUT=DROPOUT, EPOCHS=EPOCHS, PATIENCE=PATIENCE, BATCH_SIZE=BATCH_SIZE,
                     test_set=TEST_SET)

    # types = ["k_1", "k_sqrtN", "k_sqrtNDiv2", "k_sqrtNDiv4", "k_sqrtNBy2", "k_sqrtNBy4"]
    types = ["k_sqrtN", "k_sqrtNDiv2", "k_sqrtNDiv4", "k_sqrtNBy2", "k_sqrtNBy4"]
    baseline = "output_" + str(percent) + "%/performances/"
    path_baseline = "../modelling/techniques/baseline/simple_prediction/" + baseline
    if sentiment:
        path_single_target = "../modelling/techniques/forecasting/outputs_single/outputs_single_" + str(
            percent) + "/outputs_single_sentiment_" + str(
            percent) + "/single_target/result/"
        output_path = "../modelling/techniques/forecasting/outputs_multi_" + str(
            percent) + "/outputs_multi_sentiment_" + str(percent) + "/reports/"
    else:
        path_single_target = "../modelling/techniques/forecasting/outputs_single/outputs_single_" + str(
            percent) + "/outputs_single_no_sentiment_" + str(
            percent) + "/single_target/result/"
        output_path = "../modelling/techniques/forecasting/outputs_multi_" + str(
            percent) + "/outputs_multi_no_sentiment_" + str(percent) + "/reports/"
    cryptocurrencies = [["BTC", "BTS", "DASH"], ["DGB", "DOGE", "ETH"], ["IOC", "LTC",
                                                                         "MAID"], ["MONA", "NAV", "SYS"],
                        ["VTC", "XCP", "XLM"], ["XMR", "XRP"]]
    report_multi_target_crypto_oriented(path_baseline, path_single_target, types, output_path, cryptocurrencies,
                                        percent, sentiment)
    if sentiment:
        path_single_target = "../modelling/techniques/forecasting/outputs_single/outputs_single_" + str(
            percent) + "/outputs_single_sentiment_" + str(
            percent) + "/single_target/report/"
    else:
        path_single_target = "../modelling/techniques/forecasting/outputs_single/outputs_single_" + str(
            percent) + "/outputs_single_no_sentiment_" + str(
            percent) + "/single_target/report/"
    report_multi_target_k_oriented(path_single_target, types, output_path, percent, sentiment)
    if sentiment:
        path_multi_target = "../modelling/techniques/forecasting/outputs_multi_" + str(
            percent) + "/outputs_multi_sentiment_" + str(percent) + "/reports/multi_target_k_oriented/"
    else:
        path_multi_target = "../modelling/techniques/forecasting/outputs_multi_" + str(
            percent) + "/outputs_multi_no_sentiment_" + str(percent) + "/reports/multi_target_k_oriented/"
    report_multi_target_overall(path_single_target, types, output_path, path_multi_target)

    """path_multi_target = "../modelling/techniques/forecasting/"
    crypto_oriented(path_multi_target,types)"""

    # NEW: cluster oriented
    """path_multitarget="../modelling/techniques/forecasting/clusters/"
    path_singletarget = "../modelling/techniques/forecasting/output_single/result/"
    path_output="../modelling/techniques/forecasting/reports/"
    crypto_cluster_oriented(path_multitarget,path_singletarget,path_output)"""

    # NEW 2
    """path_multi_target = "../modelling/techniques/forecasting/"
    multi_vs_single(path_multi_target, types)"""

    """report_configurations(temporal_sequence=temporal_sequences, num_neurons=list_number_neurons,
                          experiment_folder=EXPERIMENT_PATH + "clusters" + "/" + cluster_n + "/",
                          results_folder="result",
                          report_folder="report", output_filename="overall_report")"""

    """report_crypto(experiment_folder=EXPERIMENT_PATH + "clusters" + "/" + cluster_n + "/",
                  result_folder="result",
                  report_folder="report",output_filename="report")"""

    # generate_line_chart(EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/",temporal_sequences,number_neurons)


def single_target_main(TEST_SET, type, features_to_use, temporal_sequences, number_neurons,
                       learning_rate, DROPOUT, EPOCHS, PATIENCE, BATCH_SIZE, percent, sentiment=0):
    #DATA_PATH = "../preparation/preprocessed_dataset_" + str(percent) + "%" + "/constructed/" + type + "/"
    DATA_PATH = "../preparation/preprocessed_dataset/constructed/" + type + "/"
    # SIMPLE PREDICTION
    OUTPUT_SIMPLE_PREDICTION = "../modelling/techniques/baseline/simple_prediction/output_" + str(percent) + "%/"
    #simple_prediction(DATA_PATH, TEST_SET, OUTPUT_SIMPLE_PREDICTION)

    # SINGLE TARGET LSTM
    if sentiment:
        EXPERIMENT_PATH = "../modelling/techniques/forecasting/outputs_single/outputs_single_" + str(
            percent) + "/outputs_single_sentiment_" + str(
            percent) + "/single_target/"
    else:
        EXPERIMENT_PATH = "../modelling/techniques/forecasting/outputs_single/outputs_single_" + str(
            percent) + "/outputs_single_no_sentiment_" + str(percent) + "/single_target/"

    TENSOR_DATA_PATH = EXPERIMENT_PATH + "tensor_data"


    single_target(EXPERIMENT_PATH=EXPERIMENT_PATH,
                  DATA_PATH=DATA_PATH,
                  TENSOR_DATA_PATH=TENSOR_DATA_PATH,
                  window_sequences=temporal_sequences,
                  list_num_neurons=number_neurons, learning_rate=learning_rate,
                  features_to_use=features_to_use,
                  DROPOUT=DROPOUT, EPOCHS=EPOCHS, PATIENCE=PATIENCE, BATCH_SIZE=BATCH_SIZE, test_set=TEST_SET)


    # visualization single_target
    input_path_single = EXPERIMENT_PATH + "result/"
    input_path_baseline = "../modelling/techniques/baseline/simple_prediction/output_" + str(
        percent) + "%/performances/"
    output_path = EXPERIMENT_PATH + "report/"
    #comparison_macro_avg_recall_single_vs_baseline(input_path_single, input_path_baseline, output_path)
    #overall_comparison_macro_avg_recall_simple_vs_baseline(input_path_single, input_path_baseline, output_path)
    #overall_macro_avg_recall_single(input_path_single, output_path)
    """report_configurations(temporal_sequence=temporal_sequences, num_neurons=number_neurons,
                          experiment_folder=EXPERIMENT_PATH, results_folder="result",
                          report_folder="report", output_filename="overall_report")"""
    '''
    report_crypto(experiment_folder=EXPERIMENT_PATH, result_folder="result", report_folder="report",
                  output_filename="report")
    # generate_line_chart(EXPERIMENT_PATH, temporal_sequences, number_neurons)
    '''

def report_single_target_sentiment_vs_no_sentiment(percent):
    input_path_single_sentiment = "../modelling/techniques/forecasting/outputs_single/outputs_single_" + str(
        percent) + "/outputs_single_sentiment_" + str(
        percent) + "/single_target/result/"
    input_path_single_no_sentiment = "../modelling/techniques/forecasting/outputs_single/outputs_single_" + str(
        percent) + "/outputs_single_no_sentiment_" + str(
        percent) + "/single_target/result/"
    input_path_baseline = "../modelling/techniques/baseline/simple_prediction/output_" + str(
        percent) + "%/performances/"
    output_path_sentiment_vs_no_sentiment = "../modelling/techniques/forecasting/outputs_single/outputs_single_" + str(
        percent) + "/report/"

    comparison_macro_avg_recall_single_sentiment_vs_single_no_sentiment(input_path_single_sentiment,
                                                                        input_path_single_no_sentiment,
                                                                        input_path_baseline,
                                                                        output_path_sentiment_vs_no_sentiment)
    overall_comparison_macro_avg_recall_single_sentiment_vs_single_no_sentiment(input_path_single_sentiment,
                                                                                input_path_single_no_sentiment,
                                                                                input_path_baseline,
                                                                                output_path_sentiment_vs_no_sentiment)


def report_multi_target_sentiment_vs_no_sentiment(percent):
    input_path_single_sentiment = "../modelling/techniques/forecasting/outputs_single/outputs_single_" + str(
        percent) + "/outputs_single_sentiment_" + str(
        percent) + "/single_target/result/"
    input_path_single_no_sentiment = "../modelling/techniques/forecasting/outputs_single/outputs_single_" + str(
        percent) + "/outputs_single_no_sentiment_" + str(
        percent) + "/single_target/result/"
    input_path_baseline = "../modelling/techniques/baseline/simple_prediction/output_" + str(
        percent) + "%/performances/"
    input_path_multi_sentiment = "../modelling/techniques/forecasting/outputs_multi_" + str(
        percent) + "/outputs_multi_sentiment_" + str(percent) + "/reports/"
    input_path_multi_no_sentiment = "../modelling/techniques/forecasting/outputs_multi_" + str(
        percent) + "/outputs_multi_no_sentiment_" + str(percent) + "/reports/"
    output_path = "../modelling/techniques/forecasting/outputs_multi_" + str(percent) + "/report/"

    folder_creator(output_path, 1)

    report_multi_target_overall_multi_sentiment_vs_no_sentiment(input_path_multi_no_sentiment,
                                                                input_path_multi_sentiment, output_path)

    cryptocurrencies = [["BTC", "BTS"], ["DASH", "DGB"], ["DOGE", "ETH"], ["IOC", "LTC"], ["MAID", "MONA"],
                        ["NAV", "SYS"],
                        ["VTC", "XCP"], ["XLM", "XMR"], ["XRP"]]

    types = ["k_sqrtN", "k_sqrtNDiv2", "k_sqrtNDiv4", "k_sqrtNBy2", "k_sqrtNBy4"]

    report_multi_target_crypto_oriented_sentiment_vs_no_sentiment(input_path_single_sentiment,
                                                                  input_path_single_no_sentiment, input_path_baseline,
                                                                  output_path, types, cryptocurrencies, percent)

    input_path_single_target = "../modelling/techniques/forecasting/outputs_single/outputs_single_" + str(
        percent) + "/report/"
    report_multi_target_k_oriented_vs_single_target(input_path_single_target,
                                                    output_path, types, percent)
    input_path_multi_target = "../modelling/techniques/forecasting/outputs_multi_" + str(
        percent) + "/report/multi_target_k_oriented/"
    report_multi_target_overall_vs_single_target(types, output_path, input_path_multi_target)


def data_understanding(crypto_names=None):
    # DATA UNDERSTANDING
    PATH_DATASET = "../acquisition/dataset/original/"

    # COLLECT INITIAL DATA
    # data collecting from yahoo finance
    get_most_important_cryptos(crypto_names, startdate=datetime(year=2018, month=2, day=2),
                               enddate=datetime(year=2021, month=6, day=1))
    get_fear_index(start_date='2018-02-01', end_date='2021-05-31')

    # EXPLORE DATA
    # missing_values(PATH_DATASET)
    # describe dataframes
    OUTPUT_PATH = "../understanding/output/"
    # describe(PATH_DATASET, OUTPUT_PATH, None, None)


"""def testing_set_baseline(interval):
    test_start_date = "2019-01-01"
    test_end_date = "2019-12-31"
    try:
        TEST_SET = get_testset(
            "../modelling/techniques/forecasting/testing/" + test_start_date + "_" + test_end_date +"_" + str(interval)+"_baseline.txt")
    except:
        generate_testset_baseline(test_start_date, test_end_date,interval, "../modelling/techniques/forecasting/testing/")
        TEST_SET = get_testset(
            "../modelling/techniques/forecasting/testing/" + test_start_date + "_" + test_end_date+"_"+ interval+"_baseline.txt")
    return TEST_SET"""


def testing_set():
    test_start_date = "2020-06-01"
    test_end_date = "2021-05-31"
    try:
        TEST_SET = get_testset(
            "../modelling/techniques/forecasting/testing/" + test_start_date + "_" + test_end_date + ".txt")
    except:
        # Test set HAS TO BE EQUAL AMONG ALL THE EXPERIMENTS!!!
        generate_testset(test_start_date, test_end_date, "../modelling/techniques/forecasting/testing/")
        TEST_SET = get_testset(
            "../modelling/techniques/forecasting/testing/" + test_start_date + "_" + test_end_date + ".txt")
    return TEST_SET


main()

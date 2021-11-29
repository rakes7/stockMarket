import ast
import os

import pandas as pd

from utility.dataset_utils import cut_dataset_by_range
from utility.reader import get_crypto_symbols_from_folder


# Dictionary
def generate_cryptocurrencies_dictionary(PATH_TO_READ, PATH_OUTPUT):
    crypto_symbols = get_crypto_symbols_from_folder(PATH_TO_READ)
    df = pd.DataFrame(columns=['id'], index=crypto_symbols)
    i = 0
    for crypto_name in crypto_symbols:
        df.at[crypto_name, "id"] = i
        i += 1
    df = df.rename_axis('symbol').reset_index()
    df.to_csv(PATH_OUTPUT + "symbol_id.csv", ",", index=False)


# selects only the datasets which cover the period of time of interest.
def prepare_dataset_for_clustering(input_path_type_for_clustering, input_path_type_for_prediction, output_path):
    for crypto in os.listdir(input_path_type_for_clustering):
        try:
            # dataset to cut
            df = cut_dataset_by_range(input_path_type_for_clustering, crypto.replace(".csv", ""), start_date, end_date)
            df = df.set_index("Date")
            if (df.index[0] == start_date):
                df = df.reset_index()
                df.to_csv(output_path + "cut_datasets/" + crypto, sep=",", index=False)
                # copyfile(input_path_type_for_prediction + crypto, output_path + "original_datasets/" + crypto)
        except:
            pass


# todo dopo cambiamo il nome
def separate_folders(CLUSTERING_PATH):
    for k_used in os.listdir(CLUSTERING_PATH + "clusters/"):
        df = pd.read_csv(CLUSTERING_PATH + "clusters/" + k_used, sep=",", header=0, index_col=0)
        for str in ast.literal_eval(df.loc[0, 'cryptos']):
            pass


# VISUALIZATION
# just joins the prediction of all cryptocurrencies. It does no average!!
def merge_predictions(experiment_folder, result_folder):
    experiment_and_result_folder = experiment_folder + result_folder + "/"

    try:
        os.remove(experiment_and_result_folder + "merged_predictions.csv")
    except:
        pass

    cryptocurrencies = os.listdir(experiment_and_result_folder)
    rows = []
    for crypto in cryptocurrencies:
        # get all the configurations
        configurations = os.listdir(experiment_and_result_folder + crypto + "/")
        configurations.sort(reverse=True)

        # for each configuration:
        for config in configurations:
            cols = ["symbol", "date", "observed_class", "predicted_class"]
            predictions_csv = pd.read_csv(
                experiment_and_result_folder + crypto + "/" + config + "/stats/predictions.csv", usecols=cols)

            # rename the columns
            predictions_csv.columns = ["symbol", "date", "observed_value", "predicted_value"]

            # number of predicted days
            length = int(len(predictions_csv['symbol']))

            # parses the configuration name (starting from the folder's name)
            conf_parsed = config.split("_")
            neurons = [str(conf_parsed[1]) for x in range(length)]
            days = [str(conf_parsed[3]) for x in range(length)]

            # inserts new columns
            predictions_csv.insert(2, "neurons", neurons, True)
            predictions_csv.insert(3, "days", days, True)

            # populate the list of values
            for row in predictions_csv.values:
                rows.append(row)

    cols = ["symbol", "date", "neurons", "days", "observed_value", "predicted_value"]
    final_csv = pd.DataFrame(rows, columns=cols)
    final_csv.to_csv(experiment_and_result_folder + "merged_predictions.csv", index=False)
    return


def generate_fileaverageRMSE_byalgorithm(path, name_final_file, experiments):
    try:
        os.remove(path + "/" + name_final_file + ".csv")
    except:
        pass
    all = {"k": [], "k_description": [], "average_rmse_norm": []}
    for experiment in experiments:
        for algorithm in os.listdir("crypto_clustering/" + experiment + "/"):
            if (
                    algorithm != "clusters" and algorithm != "cutData" and algorithm != "horizontalDataset" and algorithm != "reports"):
                # all["k"].append(len(os.listdir(path+"/"+algorithm_k))-1)
                try:
                    clusters = get_clusters2(experiment, algorithm)
                    all["k"].append(len(clusters))
                    if (experiment == "experiment_common"):
                        all["k_description"].append(algorithm.split("_")[2])
                    else:
                        name = experiment.split("_")[1]
                        all["k_description"].append(name + "_" + algorithm.split("_")[2])
                except:
                    all["k"].append(0)
                    all["k_description"].append("singleTarget")
                algorithm_results = pd.read_csv("crypto_clustering/" + experiment + "/" + algorithm + "/myresults.csv",
                                                usecols=["average_rmse_norm"])
                rmse = []
                for rmseval in algorithm_results.values:
                    rmse.append(rmseval)
                average = np.average(rmse)
                all["average_rmse_norm"].append(average)
    pd.DataFrame(all).to_csv(path + "/" + name_final_file + ".csv")


def generate_averagermseForK(path, num_of_clusters, name_experiment_model):
    # print(num_of_clusters)
    try:
        os.remove(path + "/" + "myresults.csv")
    except:
        pass
    name_folder_result = "Result"
    i = 0
    # nota:average_rmse_norm è la media di tutti gli errori per giorni e hidden neurons
    all = {"cluster_id": [], "crypto_name": [], "average_rmse_norm": []}
    while i < num_of_clusters:
        if (name_experiment_model == "MultiTarget_Data"):
            completePath = path + "/" + "cluster_" + str(
                i) + "/" + name_experiment_model + "/" + name_folder_result + "/"
        else:
            completePath = path + "/cluster_0/" + name_experiment_model + "/" + name_folder_result + "/"
        cryptocurrencies = os.listdir(completePath)
        for crypto in cryptocurrencies:
            all["cluster_id"].append(str(i))
            all["crypto_name"].append(crypto)
            configuration_used = os.listdir(completePath + crypto + "/")
            configuration_used.sort(reverse=True)
            # for each configuration:
            rmse = []
            for conf in configuration_used:
                # estra solo il valore dell'rmse
                rmseval = pd.read_csv(completePath + str(crypto) + "/" + conf + "/stats/errors.csv",
                                      usecols=["rmse_norm"]).values[0][0]
                rmse.append(rmseval)
            average_rmse = np.average(rmse)
            all["average_rmse_norm"].append(average_rmse)
        i += 1
    pd.DataFrame(all).to_csv(path + "/myresults.csv")
    return

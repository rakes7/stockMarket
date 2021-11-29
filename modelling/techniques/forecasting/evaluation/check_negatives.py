import os

import pandas as pd


def check_negatives(cluster, path):
    Result = "Result"
    file_predictions = "stats/predictions.csv"
    c1 = 0
    for id in cluster:
        for crypto in cluster[id]:
            # if "SingleTarget_Data_with_Indicators" in path: coin=coin+"_with_indicators"
            for conf in os.listdir(path + "/" + Result + "/" + crypto):
                csv = pd.read_csv(path + "/" + Result + "/" + crypto + "/" + conf + "/" + file_predictions)
                i = 0
                for value in csv["predicted_norm"]:
                    if value < 0:
                        print(csv["date"][i], csv["symbol"][i], csv["predicted_norm"][i], "(", csv["observed_norm"][i],
                              ")")
                        c1 = c1 + 1
                    i = i + 1
                print("For " + path + "/" + Result + "/" + crypto + "/" + conf + " there are: " + str(
                    c1) + " negative values")

import matplotlib.pyplot as plt
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

from utility.folder_creator import folder_creator


def describe_new(PATH_DATASET, output_path, name_folder_res=None, features_to_use=None):
    if name_folder_res == None:
        PATH_OUT = output_path + "descriptions/"
    else:
        PATH_OUT = output_path + "descriptions/" + name_folder_res + "/"

    folder_creator(PATH_OUT, 1)
    df = pd.read_csv(PATH_DATASET + "horizontal.csv", delimiter=',', header=0)
    crypto_symbol = pd.read_csv(PATH_DATASET + "symbol_id.csv", index_col=1)
    # df=cut_dataset_by_range(PATH_DATASET,crypto_name,'2017-07-20','2018-10-27')
    # if(crypto_name=="BTC"):
    PATH_CRYPTO = PATH_OUT + "horizontal/"
    folder_creator(PATH_CRYPTO + "feature_selection/", 1)
    feature_selection(df, df.columns.values, "horizontal", crypto_symbol, PATH_CRYPTO + "feature_selection/")


def feature_selection(df, features, crypto_name, crypto_symbol, output_path):
    dfnew = pd.DataFrame()
    dfnew['Date'] = df['Date']
    # dfnew=dfnew.set_index("Date")
    # leggere il file che contiene
    # fai numero + 1
    # crypto_symbol=crypto_symbol.set_index("id")
    df = df.drop("Date", axis=1)
    for f1, f2 in product(df.columns.values, df.columns.values):
        f1_splitted = f1.split("_")
        index = f1_splitted[len(f1_splitted) - 1]
        current_symbol = crypto_symbol.symbol[int(index) - 1]
        f1_replaced = f1.replace("_" + index, "")
        folder_creator(output_path + current_symbol + "/", 0)

        # folder_creator(output_path+"/", 1)
        f2_splitted = f2.split("_")
        index2 = f2_splitted[len(f2_splitted) - 1]
        f2_replaced = f2.replace("_" + index2, "")
        if f1 != f2 and f1_replaced == f2_replaced:
            second_symbol = crypto_symbol.symbol[int(index2) - 1]
            # print(f1[:-2]+"-"+f2[:-2])
            print(current_symbol + "-" + second_symbol)
            folder_creator(output_path + current_symbol + "/" + second_symbol + "/", 0)
            dfnew[str(f1)] = df[str(f1)]
            dfnew[str(f2)] = df[str(f2)]

            fig = plt.figure(figsize=(55, 8))
            ax = fig.add_subplot(1, 1, 1)
            dfnew = dfnew.set_index("Date")
            dfnew.plot(kind='line', ax=ax)

            # ax.set_xticklabels(dfnew.index.values, rotation=45, fontsize=11)
            plt.title(current_symbol + " VS " + second_symbol, fontsize=20)  # for title
            # plt.xlabel("Date", fontsize=15)  # label for x-axis
            plt.savefig(output_path + current_symbol + "/" + second_symbol + "/" + f1 + "_" + f2 + ".png", dpi=200)
            plt.clf()
            # plt.ylabel(feature, fontsize=15)  # label for y-axis
            # plt.show()
            dfnew = dfnew.reset_index()
            dfnew = dfnew.drop(f1, axis=1)
            dfnew = dfnew.drop(f2, axis=1)

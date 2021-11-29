import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import lag_plot
from scipy.stats import pearsonr, stats
from statsmodels.tsa.stattools import adfuller

from understanding.missing_values import count_missing_values, count_missing_values_by_year, \
    generate_bar_chart_by_year
from utility.folder_creator import folder_creator

COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
PATH_DATA_UNDERSTANDING = "../understanding/output/"


# check if there are missing values, generate some charts and reports about it.
def missing_values(PATH_DATASET):
    folder_creator(PATH_DATA_UNDERSTANDING, 1)
    folder_creator(PATH_DATA_UNDERSTANDING + "missing_values_by_year/", 1)
    count_missing_values(PATH_DATASET)
    count_missing_values_by_year(PATH_DATASET)
    generate_bar_chart_by_year(PATH_DATA_UNDERSTANDING + "missing_values_by_year/")


"""few statistics that give some perspective on the nature of the distribution of the data.
-count the larger this number, the more credibility all the stats have.
-mean is the average and is the "expected" value of the distribution. On average, you'd expect to get this number.
it's affected by outliers.
-std (how a set of values spread out from their mean).
A low SD shows that the values are close to the mean and a high SD shows a high diversion from the mean.
SD is affected by outliers as its calculation is based on the mean
If SD is zero, all the numbers in a dataset share the same value
-50% is also the median and it's difference from the mean gives information on the skew of the distribution. It's also another definition of average that is robust to outliers in the data.
-25% percentile is the value below which 25% of the observations may be found
-75%  percentile is the value below which 75% of the observations may be found
-min, max, max - min, 75% - 25% are all alternatives to perspectives on how big of swings the data takes relative to the mean
"""


def describe(PATH_DATASET, output_path, name_folder_res=None, features_to_use=None):
    if name_folder_res == None:
        PATH_OUT = output_path + "descriptions/"
    else:
        PATH_OUT = output_path + "descriptions/" + name_folder_res + "/"

    folder_creator(PATH_OUT, 1)
    for crypto in os.listdir(PATH_DATASET):
        crypto_name = crypto.replace(".csv", "")
        if crypto_name == "BTC":

            if features_to_use != None:
                features_to_read = features_to_use + ['Date']
                df = pd.read_csv(PATH_DATASET + crypto, delimiter=',', header=0, usecols=features_to_read)
            else:
                df = pd.read_csv(PATH_DATASET + crypto, delimiter=',', header=0)
                features_to_use = df.columns.values

            # df=cut_dataset_by_range(PATH_DATASET,crypto_name,'2017-07-20','2018-10-27')
            # if(crypto_name=="BTC"):
            PATH_CRYPTO = PATH_OUT + crypto_name + "/"
            folder_creator(PATH_CRYPTO + "general_stats/", 1)
            folder_creator(PATH_CRYPTO + "noscaling_vs_logscaling/", 1)
            folder_creator(PATH_CRYPTO + "lag_plot/", 1)
            folder_creator(PATH_CRYPTO + "box_plot/", 1)
            folder_creator(PATH_CRYPTO + "distribution_plot/", 1)
            folder_creator(PATH_CRYPTO + "correlation_heatmap/", 1)
            folder_creator(PATH_CRYPTO + "normality_test/", 1)
            folder_creator(PATH_CRYPTO + "stationary_test/", 1)
            folder_creator(PATH_CRYPTO + "feature_selection/", 1)
            folder_creator(PATH_CRYPTO + "kurtosis_test/", 1)
            folder_creator(PATH_CRYPTO + "skewness_test/", 1)

            df.describe().to_csv(PATH_CRYPTO + "general_stats/" + crypto, sep=",")

            # no_scaling_vs_log_scaling(df,features_to_use,crypto_name,PATH_CRYPTO+"noscaling_vs_logscaling/")

            feature_selection(df, features_to_use, crypto_name, PATH_CRYPTO + "feature_selection/")
            # correlation_matrix(df, crypto_name, PATH_CRYPTO + "correlation_heatmap/")
            # lag_plott(df,features_to_use,crypto_name,PATH_CRYPTO + "lag_plot/")

            box_plot(df, crypto_name, PATH_CRYPTO + "box_plot/")
    
            distribution_plot(df,features_to_use,crypto_name,PATH_CRYPTO + "distribution_plot/")
    
            stationary_test(df, features_to_use, crypto_name, PATH_CRYPTO + "stationary_test/")
    
            
    
            normality_test(df, features_to_use, crypto_name, PATH_CRYPTO + "normality_test/")

            stationary_test(df, features_to_use, crypto_name, PATH_CRYPTO + "stationary_test/")

            # kurtosis_normal_distribution(df,features_to_use,crypto_name,PATH_CRYPTO + "kurtosis_test/")

            """skewness_normal_distribution(df,features_to_use,crypto_name,PATH_CRYPTO + "skewness_test/")"""


def feature_selection(df, features, crypto_name, output_path):
    dfnew = pd.DataFrame()
    dfnew['Date'] = df['Date']
    dfnew['Close'] = df['Close']
    for feature in features:
        if feature != "Close":
            dfnew[feature] = df[feature]
            dfnew.plot()
            plt.title(feature, fontsize=20)  # for title
            plt.xlabel("Date", fontsize=15)  # label for x-axis
            plt.savefig(output_path + crypto_name + "_" + feature + ".png", dpi=150)
            plt.clf()

            # plt.ylabel(feature, fontsize=15)  # label for y-axis
            # plt.show()

            dfnew = dfnew.drop(feature, axis=1)


def skewness_normal_distribution(df, features, crypto_name, output_path):
    res = {'feature': [], 'skewness_of_n_distrib': []}
    for feature in features:
        # df = df.dropna(subset=[feature])
        stat, p = stats.skew(df[feature])
        res['feature'].append(feature)
        res['skewness_of_n_distrib'].append(stat)
    pd.DataFrame(data=res).to_csv(output_path + crypto_name + ".csv", sep=",", index=False)


def kurtosis_normal_distribution(df, features, crypto_name, output_path):
    res = {'feature': [], 'kurtosis_of_n_distrib': []}
    for feature in features:
        df = df.dropna(subset=[feature])
        stat, p = stats.kurtosis(df[feature].values)
        res['feature'].append(feature)
        res['kurtosis_of_n_distrib'].append(stat)
    pd.DataFrame(data=res).to_csv(output_path + crypto_name + ".csv", sep=",", index=False)


def normality_test(df, features, crypto_name, output_path):
    alpha = 0.05
    res = {'feature': [], 'statistics': [], 'p-value': [], 'is_gaussian': []}
    for feature in features:
        if feature != "Date":
            # trasform to be stationary
            df = df.dropna(subset=[feature])
            stat, p = stats.normaltest(df[feature])
            res['feature'].append(feature)
            res['statistics'].append(stat)
            res['p-value'].append(p)
            if p > alpha:  # fail to reject H0
                res['is_gaussian'].append('True')
            else:
                res['is_gaussian'].append('False')
    pd.DataFrame(data=res).to_csv(output_path + crypto_name + ".csv", sep=",", index=False)


def no_scaling_vs_log_scaling(df, features, crypto_name, output_path):
    for feature in features:
        # df=df.set_index('Date')
        df = df.dropna(subset=[feature])
        plt.figure(figsize=(20, 7))
        plt.subplot(1, 2, 1)
        ax = df[feature].plot(style=['-'])
        ax.lines[0].set_alpha(0.3)
        ax.set_ylim(-0.01, np.max(df[feature]))
        plt.xticks(rotation=30)
        plt.title("No scaling")
        ax.legend()

        plt.subplot(1, 2, 2)
        ax = df[feature].plot(style=['-'])
        ax.lines[0].set_alpha(0.3)
        ax.set_yscale('log')
        # ax.set_ylim(0, np.max(df['Close']+1))
        plt.xticks(rotation=30)
        plt.title("logarithmic scale")
        ax.legend()
        plt.savefig(output_path + crypto_name + "_" + feature + ".png", dpi=120)


def lag_plott(df, features, crypto_name, output_path):
    for feature in features:
        df = df.dropna(subset=[feature])
        plt.figure(figsize=(5, 5))
        plt.title("lag_plot_" + feature + "_" + crypto_name)
        lag_plot(df[feature])
        plt.savefig(output_path + crypto_name + "_" + feature + ".png", dpi=120)


def box_plot(df, crypto_name, output_path):
    plt.figure(figsize=(35, 20))
    i = 1
    for feature in df.columns.values:
        if feature != "Date":
            plt.subplot(1, len(df.columns.values) - 1, i)
            ax = sns.boxplot(x=feature, data=df, orient="v")
            ax.set_title(feature)
            i += 1
    plt.savefig(output_path + crypto_name + ".png", dpi=120)


def distribution_plot(df, features, crypto_name, output_path):
    for feature in features:
        if feature != "Date":
            filter_data = df.dropna(subset=[feature])
            plt.figure(figsize=(14, 8))
            ax = sns.distplot(filter_data[feature], kde=False)
            ax.set_title("distribution_plot_" + feature + "_" + crypto_name)
            plt.savefig(output_path + crypto_name + "_" + feature + ".png", dpi=120)


"""
In a stationary time series, statistical properties such as mean and variance are constant over time. 
In a non-stationary series, these properties are dependent on time.
Most statistical forecasting methods assume that the time series is approximately stationary. 
In a Stationary time series, there is no visible trend.
we will use the ADF test which is a type of unit root test. Unit roots are a cause for non-stationarity, the ADF test will test if unit root is present.
Null Hypothesis states there is the presence of a unit root.
Alternate Hypothesis states there is no unit root. In other words, Stationarity exists.
If statistical test is upper then 5% then we reject H0 and accept H1.


Note:Correcting for non-stationarities in an ARIMA model is necessary because it is one of the fundamental assumptions 
that you make about that particular model. LSTMs do not make that assumption. So you don't have to do it with an LSTM.
But it easier for the neural network to learn, So although you don't need to do it, it may still be a good idea and give you a boost in performance.
"""


def stationary_test(df, features, crypto_name, output_path):
    df = df.set_index("Date")
    significance_level = 0.05
    res = {'feature': [], 'adf_statistics': [], 'p-value': [], '1%': [], '5%': [], '10%': [], 'is_stationary': []}
    for feature in features:
        if feature != "Date":
            # trasform to be stationary
            df = df.dropna(subset=[feature])
            X = df[feature].values
            result = adfuller(X, autolag='AIC')
            res["adf_statistics"].append(float(result[0]))
            p_value = result[1]
            res['p-value'].append(p_value)
            res['feature'].append(feature)
            for key, value in result[4].items():
                res[key] = value
            if (p_value < significance_level):
                res['is_stationary'].append('True')
            else:
                res['is_stationary'].append('False')
    pd.DataFrame(data=res).to_csv(output_path + crypto_name + ".csv", sep=",", index=False)


def correlation_matrix(df, crypto_name, output_path):
    df = df.set_index('Date')
    plt.figure(figsize=(30, 30))
    corr = df.corr(method='spearman')
    ax = sns.heatmap(corr, annot=True)
    plt.savefig(output_path + crypto_name + ".png", dpi=120)


def bivariate_plot(df, features_name):
    threshold = 1.0
    # df=df.set_index('Date')
    d = np.array(df)
    df_transposed = np.transpose(d)

    el = np.arange(0, len(df_transposed))
    combo_index = list(itertools.product(el, repeat=2))

    fig = plt.figure(figsize=[20, 20], dpi=100, facecolor='w', edgecolor='black')
    i = 1
    for e in combo_index:
        ind1 = e.__getitem__(0)
        ind2 = e.__getitem__(1)

        correlation_coeff, p_value = pearsonr(df_transposed[ind1], df_transposed[ind2])
        title = '\n{}-{}\nCorrelation:{}\nP_value:{}'.format(features_name[ind1], features_name[ind2],
                                                             round(correlation_coeff, 10), round(p_value, 10))

        if correlation_coeff < threshold and correlation_coeff > -threshold:
            plot_correlationbtw2V(title, df_transposed[ind1], df_transposed[ind2], len(df_transposed),
                                  len(df_transposed), i, 'r*')
        else:
            plot_correlationbtw2V(title, df_transposed[ind1], df_transposed[ind2], len(df_transposed),
                                  len(df_transposed), i, 'g.')
        i = i + 1
    plt.show()
    return


def plot_correlationbtw2V(title, data1, data2, righe, colonne, indice, cm):
    plt.subplot(righe, colonne, indice)
    plt.plot(data1, data2, cm)
    # plt.tight_layout()
    plt.subplots_adjust(left=-0.2, right=0.8, top=0.8, bottom=-0.5)
    plt.title(title)
    return


def plot_boxnotch_univariateanalysis(data, features_name):
    fig2 = plt.figure(2, figsize=[10, 10], dpi=95, facecolor='w', edgecolor='black')
    numero_features = len(data)

    d = []
    for f in range(0, numero_features, 1):
        d.append(list(data[f]))
    plt.boxplot(d, notch=True)
    plt.title(f)

    fig2.show()
    plt.show()
    return


def info_univariate(data, features_name):
    df_np = np.array(data)
    df_transposed = np.transpose(d)
    for f in range(0, len(df_transposed), 1):
        ds = sorted(df_transposed[f])
        moda = stats.mode(ds)
        print('Feature: {}:\nMAX: --> {}\nMIN:  --> {}\nAVG:  --> {}\nMODE:  --> V:{} --> {}\nMed  --> {}\n'.format(
            features_name[f], np.max(df_transposed[f]),
            np.min(df_transposed[f]),
            round(np.mean(df_transposed[f]), 1),
            moda[0], moda[1],
            np.median(ds)))
    plot_boxnotch_univariateanalysis(df_transposed, features_name)
    return

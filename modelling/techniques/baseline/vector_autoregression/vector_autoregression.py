import os
import warnings
from datetime import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from modelling.techniques.forecasting.evaluation.error_measures import get_rmse
from utility.folder_creator import folder_creator

warnings.filterwarnings("ignore")


# # Utility functions
def str_to_datetime(date):
    return dt.strptime(date, '%Y-%m-%d')


def datetime_to_str(inp_dt):
    return inp_dt.strftime('%Y-%m-%d')


partial_folder = "predictions"
final_folder = "rmse"


def vector_autoregression(input_path, test_set, output_path, crypto_in_the_cluster):
    folder_creator(output_path + partial_folder + "/", 1)
    folder_creator(output_path + final_folder, 1)

    df = pd.read_csv(input_path, sep=',', header=0)
    df = df.set_index('Date')

    # it takes only "Close_X" columns (note that Date is not cut off since it is an index)
    features = df.columns
    features = [feature for feature in features if feature.startswith('Close')]
    df = df[features]

    # min max normalization, sono già normalizzati secondo me non ha senso rifare la normalizzazione
    # df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    """ print(df.head())
    print('features:', len(features))
    print('they are:', features)"""
    dataframes_out = []
    for crypto in crypto_in_the_cluster:
        df_out = pd.DataFrame(columns=["date", "observed_value", "predicted_value"])
        dataframes_out.append(df_out)

    for test_date in test_set:
        try:
            test_date = str_to_datetime(test_date)
            # get previous day (just 1)
            train_date = test_date - timedelta(days=1)

            train_date = datetime_to_str(train_date)
            test_date = datetime_to_str(test_date)
            """print('Last training day: {}'.format(train_date))
            print('Testing day: {}'.format(test_date))"""

            # splitting the dataset in test and train set, based on the date index
            df_train = df[:train_date]
            # select the row of the dataframe subject to test
            df_test = df[test_date:test_date].values[0]

            model = VAR(df_train)
            # todo scelta del miglior lag...
            """for i in [1, 2, 3, 4]:
                results = model.fit(maxlags=i)
                print('Lag Order =', i)
                print('AIC : ', results.aic)
                print('BIC : ', results.bic)
                print('FPE : ', results.fpe)
                print('HQIC: ', results.hqic, '\n')"""

            results = model.fit(maxlags=4, ic='aic')
            # get the lag order
            lag_order = results.k_ar
            # print(lag_order)
            # data to forecast, note that "values" transform a dataframe in a nparray
            # takes the last 4 elements..
            # in order to forecast, it expects up to the lag order number of observations from the past data
            data_for_forecasting = df_train.values[-lag_order:]
            # print(data_for_forecasting.shape)
            num_of_days_to_predict = 1
            y_predicted = results.forecast(data_for_forecasting, steps=num_of_days_to_predict)[0]

            # serialization, for each date
            # filename=os.path.join(output_path, partial_folder, '{}.csv'.format(test_date))
            """print(df_test)
            print(y_predicted)"""
            """df_out=pd.DataFrame()
            df_out['observed_value']=df_test
            df_out['predicted_value'] = y_predicted"""

            i = 0
            for df_out in dataframes_out:
                dataframes_out[i] = df_out.append(
                    {'date': test_date, 'observed_value': df_test[i],
                     'predicted_value': y_predicted[i]}, ignore_index=True)
                i += 1
        except Exception as e:
            print('Error, possible cause: {}'.format(e))

    i = 0
    for df_out in dataframes_out:
        df_out.to_csv(output_path + partial_folder + "/" + crypto_in_the_cluster[i] + ".csv", sep=",", index=False)
        i += 1

    # serialization
    rmses = []
    for crypto in os.listdir(output_path + partial_folder + "/"):
        df1 = pd.read_csv(output_path + partial_folder + "/" + crypto)
        # get rmse for each crypto
        rmse = get_rmse(df1['observed_value'], df1['predicted_value'])
        rmses.append(rmse)

        with open(os.path.join(output_path, final_folder, crypto.replace(".csv", "")), 'w+') as out:
            out.write(str(rmse))

    with open(os.path.join(output_path, final_folder, "average_rmse.txt"), 'w+') as out:
        final = np.mean(rmses)
        out.write(str(final))


"""  errors = []

for csv in os.listdir(os.path.join(result_folder, partial_folder)):
    res = pd.read_csv(os.path.join(result_folder, partial_folder, csv))
    error = res['Real'] - res['Predicted']
    sq_error = error ** 2
    errors.append(np.mean(sq_error))

with open(os.path.join(result_folder, final_folder, "RMSE.txt"), 'w+') as out:
    final = math.sqrt(np.mean(errors))
    out.write(str(final))"""

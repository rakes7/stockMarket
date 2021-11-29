import os
from datetime import timedelta

import numpy as np
import pandas as pd

from modelling.techniques.forecasting.evaluation.error_measures import get_classification_stats
from utility.folder_creator import folder_creator
from visualization.bar_chart.forecasting import comparison_macro_avg_recall_baseline, overall_macro_avg_recall_baseline

partial_folder = "predictions"
folder_performances = "performances"
report_folder = "reports"


def simple_prediction(data_path, test_set, result_folder):
    folder_creator(result_folder + partial_folder + "/", 1)
    folder_creator(result_folder + folder_performances, 1)
    folder_creator(result_folder + report_folder, 1)
    accuracies = []
    dict_avg_recall = {"crypto": [], "value": []}
    for crypto_name in os.listdir(data_path):
        df1 = pd.DataFrame(columns=["date", "observed_class", "predicted_class"])
        for date_to_predict in test_set:
            dataset_name = crypto_name + "_" + date_to_predict + ".csv"
            df = pd.read_csv(os.path.join(data_path, crypto_name, dataset_name), usecols=['Date', 'trend'])
            # new dataframe for output
            n_day_before = (pd.to_datetime(date_to_predict, format="%Y-%m-%d") - timedelta(days=1)).strftime('%Y-%m-%d')

            row_day_before = df[df['Date'] == n_day_before]
            row_day_before = row_day_before.set_index('Date')

            row_day_to_predict = df[df['Date'] == date_to_predict]
            row_day_to_predict = row_day_to_predict.set_index('Date')

            df1 = df1.append(
                {'date': date_to_predict, 'observed_class': row_day_to_predict.loc[date_to_predict, 'trend'],
                 'predicted_class': row_day_before.loc[n_day_before, 'trend'],
                 }, ignore_index=True)
        df1.to_csv(os.path.join(result_folder, partial_folder, crypto_name + ".csv"), sep=",", index=False)

    for crypto in os.listdir(result_folder + partial_folder + "/"):
        df = pd.read_csv(result_folder + partial_folder + "/" + crypto)
        # get rmse for each crypto
        confusion_matrix, performances = get_classification_stats(df['observed_class'], df['predicted_class'])

        dict_perf_2 = {'performance_name': [], 'value': []}
        # df_performances_2= pd.DataFrame(columns=['performance_name','value'])
        dict_perf_2['performance_name'].append("macro_avg_precision")
        dict_perf_2['value'].append(performances.get('macro avg').get('precision'))
        dict_perf_2['performance_name'].append("macro_avg_recall")
        dict_perf_2['value'].append(performances.get('macro avg').get('recall'))
        dict_perf_2['performance_name'].append("macro_avg_f1")
        dict_perf_2['value'].append(performances.get('macro avg').get('f1-score'))
        dict_perf_2['performance_name'].append("weighted_avg_precision")
        dict_perf_2['value'].append(performances.get('weighted avg').get('precision'))
        dict_perf_2['performance_name'].append("weighted_avg_recall")
        dict_perf_2['value'].append(performances.get('weighted avg').get('recall'))
        dict_perf_2['performance_name'].append("weighted_avg_f1-score")
        dict_perf_2['value'].append(performances.get('weighted avg').get('f1-score'))
        dict_perf_2['performance_name'].append("accuracy")
        dict_perf_2['value'].append(performances.get('accuracy'))
        dict_perf_2['performance_name'].append("support")
        dict_perf_2['value'].append(performances.get('weighted avg').get('support'))

        df_performances_1 = pd.DataFrame()
        z = 0
        while z < 3:
            df_performances_1 = df_performances_1.append(
                {'class': str(z),
                 'precision': performances.get(str(z)).get('precision'),
                 'recall': performances.get(str(z)).get('recall'),
                 'f1_score': performances.get(str(z)).get('f1-score'),
                 'support': performances.get(str(z)).get('support')
                 }, ignore_index=True)

            z += 1

        # serialization
        df_performances_1.to_csv(
            os.path.join(result_folder, folder_performances, crypto.replace(".csv", "_performances_part1.csv")),
            index=False)
        pd.DataFrame(dict_perf_2).to_csv(
            os.path.join(result_folder, folder_performances, crypto.replace(".csv", "_performances_part2.csv")),
            index=False)

        # 'accuracy', 'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1_score'
        accuracies.append(performances.get('macro avg').get('recall'))
        with open(os.path.join(result_folder, folder_performances, crypto.replace(".csv", "_macro_avg_recall.txt")),
                  'w+') as out:
            out.write(str(performances.get('macro avg').get('recall')))
            dict_avg_recall['crypto'].append(crypto.replace(".csv", ""))
            dict_avg_recall['value'].append(performances.get('macro avg').get('recall'))
            out.close()

        pd.DataFrame(data=confusion_matrix).to_csv(
            os.path.join(result_folder, folder_performances, crypto.replace(".csv", "_confusion_matrix.csv")),
            index=False)
    # saving all macro avg recall
    pd.DataFrame(dict_avg_recall).to_csv(
        os.path.join(result_folder, report_folder, "all_macro_avg_recall.csv"), index=False)

    # average macro avg recall
    with open(os.path.join(result_folder, folder_performances, "average_macro_avg_recall.txt"), 'w+') as out:
        final = np.mean(accuracies)
        out.write(str(final))
        out.close()

    # reports
    comparison_macro_avg_recall_baseline(
        input_file_path=os.path.join(result_folder, report_folder, "all_macro_avg_recall.csv"),
        output_path=os.path.join(result_folder, report_folder))
    overall_macro_avg_recall_baseline(
        input_file_path=os.path.join(result_folder, report_folder, "all_macro_avg_recall.csv"),
        output_path=os.path.join(result_folder, report_folder))

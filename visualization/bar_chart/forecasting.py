import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utility.folder_creator import folder_creator


def report_multi_target_overall(path_single_target, types, output_path, path_multi_target):
    df_report = pd.DataFrame()
    df = pd.read_csv(os.path.join(path_multi_target + "multi_target_k_oriented.csv"))
    avg_baseline = df.loc[df['Model'] == "baseline"]['value']
    avg_single_target = df.loc[df['Model'] == "single target"]['value']
    df_report = df_report.append({"Model": "baseline", 'value': float(avg_baseline)}, ignore_index=True)
    df_report = df_report.append({'Model': "single target", 'value': float(avg_single_target)}, ignore_index=True)
    avg_multi = []
    for k in types:
        avg_multi.append(float(df.loc[df['Model'] == k]['value']))
    avg_mu = np.average(avg_multi)
    df_report = df_report.append({'Model': "multi target", 'value': float(avg_mu)}, ignore_index=True)
    df_report.to_csv(os.path.join(output_path, "multi_target_overall.csv"), index=False)
    report_multi_target_overall_plot(df_report, output_path)


def report_multi_target_overall_plot(df, output_path):
    df.set_index('Model', inplace=True)
    df_t = df.T
    plt.figure(figsize=(10, 10))
    flatui = ["#3498db"]
    palettes = sns.color_palette(flatui)
    ax = sns.barplot(data=df_t, ci=None, palette=palettes)
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                    textcoords='offset points')
    title = "Multi-target vs single-target vs baseline overall"
    plt.title(title)
    ax.set(ylabel='Macro average recall')
    title = title.replace(" ", "_")
    plt.savefig(output_path + title + ".png", dpi=100)
    plt.close()


def report_multi_target_k_oriented(path_single_target, types, output_path, percent, sentiment):
    output_path = output_path + "multi_target_k_oriented/"
    df_report = pd.DataFrame()
    df = pd.read_csv(os.path.join(path_single_target, "single_vs_baseline_report.csv"))
    avg_baseline = df.loc[df['model type'] == "baseline"]['macro_avg_recall']
    avg_single_target = df.loc[df['model type'] == "single_target"]['macro_avg_recall']
    df_report = df_report.append({"Model": "baseline", 'value': float(avg_baseline)}, ignore_index=True)
    df_report = df_report.append({'Model': "single target", 'value': float(avg_single_target)}, ignore_index=True)
    # ora average per ogni k
    for k in types:
        if sentiment:
            path_multi_target = "../modelling/techniques/forecasting/outputs_multi_" + str(
                percent) + "/outputs_multi_sentiment_" + str(
                percent) + "/" + k + "/multi_target/"
        else:
            path_multi_target = "../modelling/techniques/forecasting/outputs_multi_" + str(
                percent) + "/outputs_multi_no_sentiment_" + str(
                percent) + "/" + k + "/multi_target/"
        highest_by_crypto = []
        for cluster in os.listdir(os.path.join(path_multi_target, "clusters/")):
            for crypto in os.listdir(os.path.join(path_multi_target, "clusters", cluster, "result")):
                highest_macro_avg_recall = -1
                for configuration in os.listdir(os.path.join(path_multi_target, "clusters", cluster, "result", crypto)):
                    df = pd.read_csv(
                        os.path.join(path_multi_target, "clusters", cluster, "result", crypto, configuration,
                                     "stats/macro_avg_recall.csv"))
                    value = df['macro_avg_recall'][0]
                    if value > highest_macro_avg_recall:
                        highest_macro_avg_recall = value
                highest_by_crypto.append(highest_macro_avg_recall)
        df_report = df_report.append({'Model': k, 'value': np.average(highest_by_crypto)}, ignore_index=True)
    folder_creator(output_path, 0)
    df_report.to_csv(os.path.join(output_path, "multi_target_k_oriented.csv"), index=False)
    report_multi_target_k_oriented_plot(df_report, output_path)


def report_multi_target_k_oriented_plot(df, output_path):
    df.set_index('Model', inplace=True)
    df_t = df.T
    plt.figure(figsize=(10, 10))
    flatui = ["#3498db"]
    palettes = sns.color_palette(flatui)
    ax = sns.barplot(data=df_t, ci=None, palette=palettes)
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                    textcoords='offset points')
    title = "Multi-target k oriented vs single-target vs baseline"
    plt.title(title)
    ax.set(ylabel='Macro average recall')
    title = title.replace(" ", "_")
    plt.savefig(output_path + title + ".png", dpi=100)
    plt.close()


def report_multi_target_crypto_oriented(path_baseline, path_single_target, types, output_path, cryptocurrencies,
                                        percent, sentiment):
    output_path = output_path + "multi_target_crypto_oriented/"
    i = 0
    report = i
    while i < len(cryptocurrencies):
        save_baseline_single_target = True
        df_report = pd.DataFrame()
        for k in types:
            if sentiment:
                path_multi_target = "../modelling/techniques/forecasting/outputs_multi_" + str(
                    percent) + "/outputs_multi_sentiment_" + str(
                    percent) + "/" + k + "/multi_target/"
            else:
                path_multi_target = "../modelling/techniques/forecasting/outputs_multi_" + str(
                    percent) + "/outputs_multi_no_sentiment_" + str(
                    percent) + "/" + k + "/multi_target/"
            for cluster in os.listdir(os.path.join(path_multi_target, "clusters/")):
                for crypto in os.listdir(os.path.join(path_multi_target, "clusters", cluster, "result")):
                    if crypto in cryptocurrencies[i]:
                        report = str(i)
                        if save_baseline_single_target:
                            file = open(path_baseline + crypto + "_macro_avg_recall.txt", "r")
                            macro_avg_recall_baseline = file.read()
                            df_report = df_report.append(
                                {'crypto': crypto, 'model type': "baseline",
                                 'macro_avg_recall': float(macro_avg_recall_baseline),
                                 'config': "standard"},
                                ignore_index=True)
                            file.close()

                            # find best for single target
                            highest_macro_avg_recall_single = -1
                            config_single = ""
                            for config_single in os.listdir(os.path.join(path_single_target, crypto)):
                                df = pd.read_csv(
                                    os.path.join(path_single_target, crypto, config_single,
                                                 "stats/macro_avg_recall.csv"),
                                    header=0)
                                if df["macro_avg_recall"][0] > highest_macro_avg_recall_single:
                                    highest_macro_avg_recall_single = df["macro_avg_recall"][0]
                                    config_single = config_single
                            df_report = df_report.append(
                                {'crypto': crypto, 'model type': "single_target",
                                 'macro_avg_recall': float(highest_macro_avg_recall_single),
                                 'config': config_single},
                                ignore_index=True)
                        highest_macro_avg_recall = -1
                        best_conf = ""
                        for configuration in os.listdir(
                                os.path.join(path_multi_target, "clusters", cluster, "result", crypto)):
                            df = pd.read_csv(
                                os.path.join(path_multi_target, "clusters", cluster, "result", crypto, configuration,
                                             "stats/macro_avg_recall.csv"))
                            value = df['macro_avg_recall'][0]
                            if value > highest_macro_avg_recall:
                                highest_macro_avg_recall = value
                                best_conf = configuration
                        df_report = df_report.append(
                            {'crypto': crypto, 'model type': k,
                             'macro_avg_recall': float(highest_macro_avg_recall),
                             'config': best_conf},
                            ignore_index=True)
            save_baseline_single_target = False
        i += 1
        folder_creator(output_path, 0)
        df_report.to_csv(os.path.join(output_path, "multi_target_crypto_oriented_" + report + ".csv"), index=False)
        report_multi_target_crypto_oriented_plot(df_report, output_path, report)


def report_multi_target_crypto_oriented_plot(df, output_path, report):
    title = "Multi-target crypto oriented results"
    fig = plt.figure(figsize=(15, 10))
    flatui = ["#ec7063", "#a569bd", "#5dade2", "#27ae60", "#f4d03f", "#f39c12", "#b2babb", "#138d75"]
    palettes = sns.color_palette(flatui)
    # sns.set(font_scale=0.65, style='white')
    ax = sns.barplot(x="crypto", y="macro_avg_recall", hue="model type", data=df, palette=palettes)
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=8, color='black', xytext=(0, 5),
                    textcoords='offset points')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
    # Put a legend to the right side
    ax.legend(loc='center right', bbox_to_anchor=(1.16, 0.5), ncol=1)
    plt.title(title)
    title = title.replace(" ", "_")
    ax.set(xlabel='Cryptocurrencies', ylabel='Macro average recall')
    plt.savefig(output_path + "/" + title + "_" + report + ".png", dpi=100)
    plt.close(fig)


def report_multi_target_k_oriented_vs_single_target(input_path_single_target,
                                                    output_path, types, percent):
    output_path = output_path + "multi_target_k_oriented/"
    df_report = pd.DataFrame()

    df = pd.read_csv(os.path.join(input_path_single_target, "sentiment_vs_no_sentiment_report.csv"))

    avg_baseline = df.loc[df['model type'] == "baseline"]['macro_avg_recall']
    avg_single_no_sentiment = df.loc[df['model type'] == "no_sentiment"]['macro_avg_recall']
    avg_single_sentiment = df.loc[df['model type'] == "sentiment"]['macro_avg_recall']
    df_report = df_report.append({"Model": "baseline", 'value': float(avg_baseline)}, ignore_index=True)
    df_report = df_report.append({'Model': "single target no sentiment", 'value': float(avg_single_no_sentiment)},
                                 ignore_index=True)
    df_report = df_report.append({'Model': "single target sentiment", 'value': float(avg_single_sentiment)},
                                 ignore_index=True)

    # ora average per ogni k no sentiment
    for k in types:
        path_multi_target = "../modelling/techniques/forecasting/outputs_multi_" + str(
            percent) + "/outputs_multi_no_sentiment_" + str(
            percent) + "/" + k + "/multi_target/"
        highest_by_crypto = []
        for cluster in os.listdir(os.path.join(path_multi_target, "clusters/")):
            for crypto in os.listdir(os.path.join(path_multi_target, "clusters", cluster, "result")):
                highest_macro_avg_recall = -1
                for configuration in os.listdir(os.path.join(path_multi_target, "clusters", cluster, "result", crypto)):
                    df = pd.read_csv(
                        os.path.join(path_multi_target, "clusters", cluster, "result", crypto, configuration,
                                     "stats/macro_avg_recall.csv"))
                    value = df['macro_avg_recall'][0]
                    if value > highest_macro_avg_recall:
                        highest_macro_avg_recall = value
                highest_by_crypto.append(highest_macro_avg_recall)
        df_report = df_report.append({'Model': k + " no sentiment", 'value': np.average(highest_by_crypto)},
                                     ignore_index=True)

    # ora average per ogni k sentiment
    for k in types:
        path_multi_target = "../modelling/techniques/forecasting/outputs_multi_" + str(
            percent) + "/outputs_multi_sentiment_" + str(
            percent) + "/" + k + "/multi_target/"
        highest_by_crypto = []
        for cluster in os.listdir(os.path.join(path_multi_target, "clusters/")):
            for crypto in os.listdir(os.path.join(path_multi_target, "clusters", cluster, "result")):
                highest_macro_avg_recall = -1
                for configuration in os.listdir(
                        os.path.join(path_multi_target, "clusters", cluster, "result", crypto)):
                    df = pd.read_csv(
                        os.path.join(path_multi_target, "clusters", cluster, "result", crypto, configuration,
                                     "stats/macro_avg_recall.csv"))
                    value = df['macro_avg_recall'][0]
                    if value > highest_macro_avg_recall:
                        highest_macro_avg_recall = value
                highest_by_crypto.append(highest_macro_avg_recall)
        df_report = df_report.append({'Model': k + " sentiment", 'value': np.average(highest_by_crypto)},
                                     ignore_index=True)

    folder_creator(output_path, 0)
    df_report.to_csv(os.path.join(output_path, "multi_target_k_oriented.csv"), index=False)
    report_multi_target_k_oriented_plot_vs_single_target(df_report, output_path)


def report_multi_target_k_oriented_plot_vs_single_target(df, output_path):
    df.set_index('Model', inplace=True)
    df_t = df.T
    plt.figure(figsize=(10, 5))
    flatui = ["#ec7063", "#a569bd", "#5dade2", "#27ae60", "#f4d03f", "#f39c12", "#b2babb", "#138d75", "#cc66ff",
              "#4dff4d", "#59b300", "#0000b3", "#ff0000"]
    palettes = sns.color_palette(flatui)
    ax = sns.barplot(data=df_t, ci=None, palette=palettes)
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    title = "Multi-target k oriented vs single-target vs baseline"
    plt.title(title)
    ax.set(ylabel='Macro average recall')
    plt.xticks(ha="right", rotation=45)
    title = title.replace(" ", "_")
    plt.gcf().subplots_adjust(bottom=0.20)
    plt.savefig(output_path + title + ".png", dpi=100, bbox_inches='tight')
    plt.close()


def report_multi_target_overall_vs_single_target(types, output_path, path_multi_target):
    df_report = pd.DataFrame()
    df = pd.read_csv(os.path.join(path_multi_target + "multi_target_k_oriented.csv"))

    avg_baseline = df.loc[df['Model'] == "baseline"]['value']
    avg_single_no_sentiment = df.loc[df['Model'] == "single target no sentiment"]['value']
    avg_single_sentiment = df.loc[df['Model'] == "single target sentiment"]['value']
    df_report = df_report.append({"Model": "baseline", 'value': float(avg_baseline)}, ignore_index=True)
    df_report = df_report.append({'Model': "single target no sentiment", 'value': float(avg_single_no_sentiment)},
                                 ignore_index=True)
    df_report = df_report.append({'Model': "single target sentiment", 'value': float(avg_single_sentiment)},
                                 ignore_index=True)
    avg_multi_no_sentiment = []
    avg_multi_sentiment = []

    for k in types:
        avg_multi_no_sentiment.append(float(df.loc[df['Model'] == k + " no sentiment"]['value']))
    avg_mu_no_sentiment = np.average(avg_multi_no_sentiment)

    for k in types:
        avg_multi_sentiment.append(float(df.loc[df['Model'] == k + " no sentiment"]['value']))
    avg_mu_sentiment = np.average(avg_multi_sentiment)

    df_report = df_report.append({'Model': "multi target no sentiment", 'value': float(avg_mu_no_sentiment)},
                                 ignore_index=True)
    df_report = df_report.append({'Model': "multi target sentiment", 'value': float(avg_mu_sentiment)},
                                 ignore_index=True)
    df_report.to_csv(os.path.join(output_path, "multi_target_overall.csv"), index=False)
    report_multi_target_overall_plot_vs_single_target(df_report, output_path)


def report_multi_target_overall_plot_vs_single_target(df, output_path):
    df.set_index('Model', inplace=True)
    df_t = df.T
    plt.figure(figsize=(5, 5))
    flatui = ["#3498db", "#FF9633", "#ff0000", "#FFD700", "#008800"]
    palettes = sns.color_palette(flatui)
    ax = sns.barplot(data=df_t, ci=None, palette=palettes)
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    title = "Multi-target vs single-target vs baseline overall"
    plt.title(title, fontsize=9)
    plt.xticks(ha="right", rotation=30, fontsize=9)
    ax.set(ylabel='Macro average recall')
    title = title.replace(" ", "_")
    plt.gcf().subplots_adjust(bottom=0.20)
    plt.savefig(output_path + title + ".png", dpi=100, bbox_inches='tight')
    plt.close()


def overall_macro_avg_recall_baseline(input_file_path, output_path):
    df = pd.read_csv(input_file_path, header=0)
    avg = np.average(df.value)
    df = pd.DataFrame(columns=['All cryptocurrencies'])
    df = df.append({'All cryptocurrencies': avg}, ignore_index=True)
    plt.figure(figsize=(5, 5))
    flatui = ["#3498db"]
    palettes = sns.color_palette(flatui)
    ax = sns.barplot(data=df, ci=None, palette=palettes)
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                    textcoords='offset points')
    title = "Baseline overall"
    plt.title(title)
    ax.set(ylabel='Average macro average recall')
    plt.savefig(output_path + "/overall_" + title + ".png", dpi=100)
    plt.close()


def comparison_macro_avg_recall_baseline(input_file_path, output_path):
    df = pd.read_csv(input_file_path, header=0)
    df.set_index('crypto', inplace=True)
    df_t = df.T
    dfs = []
    dfs.append(df_t.iloc[:, :9])
    dfs.append(df_t.iloc[:, 9:18])

    i = 1
    for df in dfs:
        plt.figure(figsize=(10, 10))
        flatui = ["#3498db"]
        palettes = sns.color_palette(flatui)
        # sns.set(font_scale=0.80, style="white")
        ax = sns.barplot(data=df, ci=None, palette=palettes)
        # annotate axis = seaborn axis
        for p in ax.patches:
            ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                        textcoords='offset points')
        # _ = ax.set_ylim(0, 5)  # To make space for the annotations
        title = "Baseline"
        plt.title(title)
        ax.set(xlabel='Cryptocurrencies', ylabel='Macro average recall')
        plt.savefig(output_path + "/comparison_" + title + "_" + str(i) + ".png", dpi=200, bbox_inches='tight')
        plt.close()
        i += 1


# input path single: path to results folder
# input path_baseline: performances
# output_path: report for each crypto
def comparison_macro_avg_recall_single_vs_baseline(input_path_single, input_path_baseline, output_path):
    folder_creator(output_path, 0)
    df_report = pd.DataFrame()
    for crypto in os.listdir(input_path_single):
        # read baseline
        file = open(input_path_baseline + crypto + "_macro_avg_recall.txt", "r")
        macro_avg_recall_baseline = file.read()
        file.close()

        # find best configuration
        max_macro_avg_recall = -1
        config = ""
        for configuration in os.listdir(os.path.join(input_path_single, crypto)):
            df = pd.read_csv(os.path.join(input_path_single, crypto, configuration, "stats/macro_avg_recall.csv"),
                             header=0)
            if df["macro_avg_recall"][0] > max_macro_avg_recall:
                max_macro_avg_recall = df["macro_avg_recall"][0]
                config = configuration

        # generate csv containing these info
        df_report = df_report.append(
            {'crypto': crypto, 'model type': 'single_target', 'macro_avg_recall': float(max_macro_avg_recall),
             'config': config},
            ignore_index=True)
        df_report = df_report.append(
            {'crypto': crypto, 'model type': 'baseline', 'macro_avg_recall': float(macro_avg_recall_baseline),
             'config': 'standard'}, ignore_index=True)
    df_report.to_csv(os.path.join(output_path, "single_target_vs_baseline_report.csv"), index=False)
    comparison_macro_avg_recall_single_vs_baseline_plot(df_report, output_path)


def comparison_macro_avg_recall_single_vs_baseline_plot(df, output_path):
    dfs = []
    dfs.append(df.iloc[:12, :])
    dfs.append(df.iloc[12:24, :])
    dfs.append(df.iloc[24:36, :])

    i = 1
    for df_l in dfs:
        title = "Single target VS Baseline"
        fig = plt.figure(figsize=(10, 10))
        flatui = ["#3498db", "#FF9633"]
        palettes = sns.color_palette(flatui)
        # sns.set(font_scale=0.65, style='white')
        ax = sns.barplot(x="crypto", y="macro_avg_recall", hue="model type", data=df_l, palette=palettes)
        for p in ax.patches:
            ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=8, color='black', xytext=(0, 5),
                        textcoords='offset points')
        plt.title(title)
        # Removed 'ax' from T.W.'s answer here aswell:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
        # Put a legend to the right side
        ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
        title = title.replace(" ", "_")
        ax.set(xlabel='Cryptocurrencies', ylabel='Macro average recall')
        plt.savefig(output_path + "/" + title + "_" + str(i) + ".png", dpi=100, bbox_inches='tight')
        plt.close(fig)
        i += 1


def overall_macro_avg_recall_single(input_path_single, output_path):
    avg_bests = []
    for crypto in os.listdir(input_path_single):
        # find best configuration
        max_macro_avg_recall = -1
        config = ""
        for configuration in os.listdir(os.path.join(input_path_single, crypto)):
            df = pd.read_csv(os.path.join(input_path_single, crypto, configuration, "stats/macro_avg_recall.csv"),
                             header=0)
            if df["macro_avg_recall"][0] > max_macro_avg_recall:
                max_macro_avg_recall = df["macro_avg_recall"][0]
                config = configuration
        avg_bests.append(max_macro_avg_recall)

    avg_single_target = np.average(avg_bests)
    df_report = pd.DataFrame()
    df_report = df_report.append(
        {'crypto': "All cryptocurrencies", 'model type': 'single_target', 'macro_avg_recall': float(avg_single_target)},
        ignore_index=True)
    df_report.to_csv(os.path.join(output_path, "single_average_report.csv"), index=False)
    overall_macro_avg_recall_single_plot(df_report, output_path)


def overall_macro_avg_recall_single_plot(df, output_path):
    title = "Single target overall"
    plt.figure(figsize=(5, 5))
    flatui = ["#3498db", "#FF9633"]
    palettes = sns.color_palette(flatui)
    ax = sns.barplot(x="crypto", y="macro_avg_recall", data=df, palette=palettes)
    plt.title(title)
    ax.set(xlabel=" ", ylabel='Average macro average recall')
    title = title.replace(" ", "_")
    plt.savefig(output_path + "/" + title + ".png", dpi=100)
    plt.close()


def overall_comparison_macro_avg_recall_simple_vs_baseline(input_path_single, input_path_baseline, output_path):
    avg_bests = []
    for crypto in os.listdir(input_path_single):

        # find best configuration
        max_macro_avg_recall = -1
        config = ""
        for configuration in os.listdir(os.path.join(input_path_single, crypto)):
            df = pd.read_csv(os.path.join(input_path_single, crypto, configuration, "stats/macro_avg_recall.csv"),
                             header=0)
            if df["macro_avg_recall"][0] > max_macro_avg_recall:
                max_macro_avg_recall = df["macro_avg_recall"][0]
                config = configuration
        avg_bests.append(max_macro_avg_recall)

    # average baseline
    file = open(input_path_baseline + "average_macro_avg_recall.txt", "r")
    avg_macro_avg_recall_baseline = file.read()
    file.close()
    avg_single_target = np.average(avg_bests)
    df_report = pd.DataFrame()
    df_report = df_report.append(
        {'crypto': "All cryptocurrencies", 'model type': 'single_target', 'macro_avg_recall': float(avg_single_target)},
        ignore_index=True)
    df_report = df_report.append(
        {'crypto': "All cryptocurrencies", 'model type': 'baseline',
         'macro_avg_recall': float(avg_macro_avg_recall_baseline)},
        ignore_index=True)
    df_report.to_csv(os.path.join(output_path, "single_vs_baseline_report.csv"), index=False)
    overall_comparison_macro_avg_recall_simple_vs_baseline_plot(df_report, output_path)


def overall_comparison_macro_avg_recall_simple_vs_baseline_plot(df, output_path):
    title = "Single target VS Baseline overall"
    fig = plt.figure(figsize=(7, 7))
    flatui = ["#3498db", "#FF9633"]
    palettes = sns.color_palette(flatui)
    ax = sns.barplot(x="crypto", y="macro_avg_recall", hue="model type", data=df, palette=palettes)
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
    # Put a legend to the right side
    ax.legend(loc='center right', bbox_to_anchor=(1.33, 0.5), ncol=1)
    plt.title(title)
    ax.set(xlabel=" ", ylabel='Average macro average recall')
    title = title.replace(" ", "_")
    plt.savefig(output_path + "/" + title + ".png", dpi=100)
    plt.close(fig)


# UNUSED
# for forecasting
def report_configurations(temporal_sequence, num_neurons, experiment_folder,
                          results_folder, report_folder, output_filename):
    # Folder creator
    experiment_and_report_folder = experiment_folder + report_folder + "/"
    experiment_and_result_folder = experiment_folder + results_folder + "/"
    folder_creator(experiment_and_report_folder, 1)

    kind_of_report = "configurations_oriented"
    folder_creator(experiment_and_report_folder + kind_of_report + "/", 1)

    # read cryptocurrencies
    cryptocurrencies = os.listdir(experiment_and_result_folder)

    # create dictionary for overall output
    # overall_report = {'model': [], 'mean_rmse_norm': [], 'mean_rmse_denorm': []}
    overall_report = {'model': [], 'mean_accuracy': []}
    overall_report['model'].append("simple prediction model")
    OUTPUT_SIMPLE_PREDICTION = "../modelling/techniques/baseline/simple_prediction/output/"
    file = open(OUTPUT_SIMPLE_PREDICTION + "average_accuracy/average_accuracy.txt", "r")
    value1 = file.read()
    file.close()
    overall_report['mean_accuracy'].append(value1)
    for window, num_neurons in product(temporal_sequence, num_neurons):
        configuration = "LSTM_{}_neurons_{}_days".format(num_neurons, window)

        # model_report = {'crypto_symbol': [], 'rmse_list_norm': [], 'rmse_list_denorm': []}
        model_report = {'crypto_symbol': [], 'accuracy_list': []}
        for crypto in cryptocurrencies:
            # read the error files of a specific crypto
            accuracy_file = pd.read_csv(
                experiment_and_result_folder + crypto + "/" + configuration + "/stats/macro_avg_recall.csv",
                index_col=0, sep=',')

            # populate the dictionary
            model_report['crypto_symbol'].append(crypto)
            model_report['accuracy_list'].append(accuracy_file["macro_avg_recall"])
            # model_report['rmse_list_denorm'].append(errors_file["rmse_denorm"])

        # Folder creator
        folder_creator(experiment_and_report_folder + kind_of_report + "/" + configuration + "/", 0)

        average_macro_avg_recall = np.mean(model_report['accuracy_list'])
        # average_rmse_denormalized = np.mean(model_report['rmse_list_denorm'])

        # configuration_report = {"Average_RMSE_norm": [], "Average_RMSE_denorm": []}
        configuration_report = {"Average_macro_avg_recall": []}
        configuration_report["Average_macro_avg_recall"].append(average_macro_avg_recall)
        # configuration_report["Average_RMSE_denorm"].append(average_rmse_denormalized)

        pd.DataFrame(configuration_report).to_csv(
            experiment_and_report_folder + kind_of_report + "/" + configuration + "/report.csv", index=False)

        # populate overall report
        overall_report['model'].append(configuration)
        overall_report['mean_accuracy'].append(average_macro_avg_recall)
        # overall_report['mean_rmse_denorm'].append(average_rmse_denormalized)

    # overall report to dataframe
    pd.DataFrame(overall_report).to_csv(
        experiment_and_report_folder + kind_of_report + "/" + output_filename + ".csv", index=False)

    # plot overall report
    plot_report(
        path_file=experiment_and_report_folder + kind_of_report + "/" + output_filename + ".csv",
        x_data="model", column_of_data="mean_accuracy", label_for_values_column="Macro avg recall",
        label_x="Configurations", title_img="Macro avg recall - Configurations Oriented",
        destination=experiment_and_report_folder + kind_of_report + "/",
        name_file_output="bargraph_accuracy_configurations_oriented")

    return


# for forecasting
def report_crypto(experiment_folder, result_folder, report_folder, output_filename):
    kind_of_report = "crypto_oriented"

    # Folder creator
    experiment_and_report_folder = experiment_folder + report_folder + "/"
    experiment_and_result_folder = experiment_folder + result_folder + "/"
    folder_creator(experiment_and_report_folder, 0)
    folder_creator(experiment_and_report_folder + kind_of_report + "/", 0)

    # read cryptocurrencies
    cryptocurrencies = os.listdir(experiment_and_result_folder)

    # for each crypto:
    for crypto in cryptocurrencies:
        # folder creator
        CRYPTO_FOLDER_PATH = experiment_and_report_folder + kind_of_report + "/" + crypto + "/"
        folder_creator(CRYPTO_FOLDER_PATH, 1)

        # dictionary for report
        # report_dic = {'configuration': [], 'RMSE_normalized': [], 'RMSE_denormalized': []}
        report_dic = {'configuration': [], 'Accuracy': []}
        # get the configurations used by the name of their folder
        configurations = os.listdir(experiment_and_result_folder + crypto + "/")
        configurations.sort(reverse=True)

        # for each configuration
        for configuration in configurations:
            # save the configuration's name in the dictionary
            report_dic['configuration'].append(configuration)

            # read 'predictions.csv' file
            macro_avg_recall_file = pd.read_csv(
                experiment_and_result_folder + crypto + "/" + configuration + "/stats/macro_avg_recall.csv")

            # get the mean of the rmse (normalized)
            avg_accuracy = macro_avg_recall_file["macro_avg_recall"].mean()
            # save in the dictionary
            report_dic['Accuracy'].append(float(avg_accuracy))

            # get the mean of the rmse (denormalized)
            # avg_rmse_denorm = errors_file["rmse_denorm"].mean()
            # save in the dictionary
            # report_dic['RMSE_denormalized'].append(float(avg_rmse_denorm))

        # save as '.csv' the dictionary in CRYPTO_FOLDER_PATH
        pd.DataFrame(report_dic).to_csv(
            experiment_and_report_folder + kind_of_report + "/" + crypto + "/" + output_filename + ".csv", index=False)

        plot_report(
            path_file=experiment_and_report_folder + kind_of_report + "/" + crypto + "/" + output_filename + ".csv",
            x_data="configuration", column_of_data="Accuracy", label_for_values_column="Macro avg recall (average)",
            label_x="Configurations", title_img="Average Macro avg recall - " + str(crypto),
            destination=experiment_and_report_folder + kind_of_report + "/" + crypto + "/",
            name_file_output="bargraph_macro_avg_recall_" + str(crypto))
    return


# for forecasting
def plot_report(path_file, x_data, column_of_data, label_for_values_column, label_x, title_img, destination,
                name_file_output):
    # read the report
    report_csv = pd.read_csv(path_file, header=0)

    # read some columns
    configurations = report_csv[x_data]
    mean_rmse_normalized = report_csv[column_of_data]

    # define the index
    index = np.arange(len(configurations))

    # create figure
    f = plt.figure()

    # bar chart
    plt.bar(index, mean_rmse_normalized)

    # set the labels
    plt.ylabel(label_for_values_column, fontsize=10)
    plt.xlabel(label_x, fontsize=10)
    plt.title(title_img)

    # customize the labels by rotating of 90 degree, for example
    plt.xticks(index, configurations, fontsize=7, rotation=90)

    # serialization
    f.savefig(destination + name_file_output, bbox_inches='tight', pad_inches=0, dpi=120)
    return


# input path single_sentiment: path to results folder
# input path_single_no_sentiment: path to result folder
# output_path: report for each crypto
def comparison_macro_avg_recall_single_sentiment_vs_single_no_sentiment(input_path_single_sentiment,
                                                                        input_path_single_no_sentiment,
                                                                        input_path_baseline,
                                                                        output_path):
    folder_creator(output_path, 0)
    df_report = pd.DataFrame()
    cryptos = []

    for crypto in os.listdir(input_path_single_sentiment):
        cryptos.append(str(crypto))

        # read baseline
        file = open(input_path_baseline + crypto + "_macro_avg_recall.txt", "r")
        macro_avg_recall_baseline = file.read()
        file.close()

        # find best configuration no sentiment
        max_macro_avg_recall_no_sentiment = -1
        config_no_sentiment = ""
        for configuration in os.listdir(os.path.join(input_path_single_no_sentiment, crypto)):
            df = pd.read_csv(
                os.path.join(input_path_single_no_sentiment, crypto, configuration, "stats/macro_avg_recall.csv"),
                header=0)
            if df["macro_avg_recall"][0] > max_macro_avg_recall_no_sentiment:
                max_macro_avg_recall_no_sentiment = df["macro_avg_recall"][0]
                config_no_sentiment = configuration

        # find best configuration sentiment
        max_macro_avg_recall_sentiment = -1
        config_sentiment = ""
        for configuration in os.listdir(os.path.join(input_path_single_sentiment, crypto)):
            df = pd.read_csv(
                os.path.join(input_path_single_sentiment, crypto, configuration, "stats/macro_avg_recall.csv"),
                header=0)
            if df["macro_avg_recall"][0] > max_macro_avg_recall_sentiment:
                max_macro_avg_recall_sentiment = df["macro_avg_recall"][0]
                config_sentiment = configuration

        # generate csv containing these info
        df_report = df_report.append(
            {'crypto': crypto, 'model type': 'baseline', 'macro_avg_recall': float(macro_avg_recall_baseline),
             'config': 'standard'}, ignore_index=True)
        df_report = df_report.append(
            {'crypto': crypto, 'model type': 'sentiment',
             'macro_avg_recall': float(max_macro_avg_recall_sentiment),
             'config': config_sentiment},
            ignore_index=True)
        df_report = df_report.append(
            {'crypto': crypto, 'model type': 'no_sentiment',
             'macro_avg_recall': float(max_macro_avg_recall_no_sentiment),
             'config': config_no_sentiment}, ignore_index=True)
    df_report.to_csv(os.path.join(output_path, "single_target_sentiment_vs_no_sentiment_report.csv"), index=False)
    comparison_macro_avg_recall_single_sentiment_vs_single_no_sentiment_plot(df_report, output_path, cryptos)


def comparison_macro_avg_recall_single_sentiment_vs_single_no_sentiment_plot(df, output_path, cryptos):
    dfs = []
    dfs.append(df.iloc[:15, :])
    dfs.append(df.iloc[15:30, :])
    dfs.append(df.iloc[30:45, :])
    dfs.append(df.iloc[45:51, :])
    i = 1
    for df_l in dfs:
        title = "Single target VS Baseline"
        fig = plt.figure(figsize=(10, 5))
        flatui = ["#3498db", "#FF9633", "#ff0000"]
        palettes = sns.color_palette(flatui)
        # sns.set(font_scale=0.65, style='white')
        ax = sns.barplot(x="crypto", y="macro_avg_recall", hue="model type", data=df_l, palette=palettes)
        for p in ax.patches:
            ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                        textcoords='offset points')
        plt.title(title)
        # Removed 'ax' from T.W.'s answer here aswell:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
        # Put a legend to the right side
        ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
        title = title.replace(" ", "_")
        ax.set(xlabel='Cryptocurrencies', ylabel='Macro average recall')
        plt.savefig(output_path + "/" + title + "_" + str(i) + ".png", dpi=100, bbox_inches='tight')
        plt.close(fig)
        i += 1


def overall_comparison_macro_avg_recall_single_sentiment_vs_single_no_sentiment(input_path_single_sentiment,
                                                                                input_path_single_no_sentiment,
                                                                                input_path_baseline,
                                                                                output_path):
    avg_bests_sentiment = []
    avg_bests_no_sentiment = []

    # average baseline
    file = open(input_path_baseline + "average_macro_avg_recall.txt", "r")
    avg_macro_avg_recall_baseline = file.read()
    file.close()

    for crypto in os.listdir(input_path_single_sentiment):

        # find best configuration sentiment
        max_macro_avg_recall_sentiment = -1
        config_sentiment = ""
        for configuration in os.listdir(os.path.join(input_path_single_sentiment, crypto)):
            df = pd.read_csv(
                os.path.join(input_path_single_sentiment, crypto, configuration, "stats/macro_avg_recall.csv"),
                header=0)
            if df["macro_avg_recall"][0] > max_macro_avg_recall_sentiment:
                max_macro_avg_recall_sentiment = df["macro_avg_recall"][0]
                config_sentiment = configuration
        avg_bests_sentiment.append(max_macro_avg_recall_sentiment)
    avg_sentiment = np.average(avg_bests_sentiment)

    for crypto in os.listdir(input_path_single_no_sentiment):

        # find best configuration no sentiment
        max_macro_avg_recall_no_sentiment = -1
        config_no_sentiment = ""
        for configuration in os.listdir(os.path.join(input_path_single_no_sentiment, crypto)):
            df = pd.read_csv(
                os.path.join(input_path_single_no_sentiment, crypto, configuration, "stats/macro_avg_recall.csv"),
                header=0)
            if df["macro_avg_recall"][0] > max_macro_avg_recall_no_sentiment:
                max_macro_avg_recall_no_sentiment = df["macro_avg_recall"][0]
                config_no_sentiment = configuration
        avg_bests_no_sentiment.append(max_macro_avg_recall_no_sentiment)
    avg_no_sentiment = np.average(avg_bests_no_sentiment)

    df_report = pd.DataFrame()
    df_report = df_report.append(
        {'crypto': "All cryptocurrencies", 'model type': 'baseline',
         'macro_avg_recall': float(avg_macro_avg_recall_baseline)},
        ignore_index=True)
    df_report = df_report.append(
        {'crypto': "All cryptocurrencies", 'model type': 'sentiment', 'macro_avg_recall': float(avg_sentiment)},
        ignore_index=True)
    df_report = df_report.append(
        {'crypto': "All cryptocurrencies", 'model type': 'no_sentiment',
         'macro_avg_recall': float(avg_no_sentiment)},
        ignore_index=True)

    df_report.to_csv(os.path.join(output_path, "sentiment_vs_no_sentiment_report.csv"), index=False)
    overall_comparison_macro_avg_recall_single_sentiment_vs_single_no_sentiment_plot(df_report, output_path)


def overall_comparison_macro_avg_recall_single_sentiment_vs_single_no_sentiment_plot(df, output_path):
    title = "Single target VS Baseline"
    fig = plt.figure(figsize=(5, 5))
    flatui = ["#3498db", "#FF9633", "#ff0000"]
    palettes = sns.color_palette(flatui)
    ax = sns.barplot(x="crypto", y="macro_avg_recall", hue="model type", data=df, palette=palettes)
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
    # Put a legend to the right side
    ax.legend(loc='center right', bbox_to_anchor=(1.33, 0.5), ncol=1)
    plt.title(title)
    ax.set(xlabel=" ", ylabel='Average macro average recall')
    title = title.replace(" ", "_")
    plt.savefig(output_path + "/" + title + ".png", dpi=100, bbox_inches='tight')
    plt.close(fig)


def report_multi_target_overall_multi_sentiment_vs_no_sentiment(path_multi_target_no_sentiment,
                                                                path_multi_target_sentiment, output_path):
    df_report = pd.DataFrame()
    df_no_sentiment = pd.read_csv(os.path.join(path_multi_target_no_sentiment + "multi_target_overall.csv"))
    df_sentiment = pd.read_csv(os.path.join(path_multi_target_sentiment + "multi_target_overall.csv"))
    overall_no_sentiment = df_no_sentiment.loc[df_no_sentiment['Model'] == "multi target"]['value']
    overall_sentiment = df_sentiment.loc[df_no_sentiment['Model'] == "multi target"]['value']
    df_report = df_report.append({'Model': "multi target no sentiment", 'value': float(overall_no_sentiment)},
                                 ignore_index=True)
    df_report = df_report.append({'Model': "multi target sentiment", 'value': float(overall_sentiment)},
                                 ignore_index=True)
    df_report.to_csv(os.path.join(output_path, "multi_target_overall_sentiment_vs_no_sentiment.csv"), index=False)
    report_multi_target_overall_plot_multi_sentiment_vs_no_sentiment(df_report, output_path)


def report_multi_target_overall_plot_multi_sentiment_vs_no_sentiment(df, output_path):
    df.set_index('Model', inplace=True)
    df_t = df.T
    plt.figure(figsize=(10, 10))
    flatui = ["#3498db", "#FF9633"]
    palettes = sns.color_palette(flatui)
    ax = sns.barplot(data=df_t, ci=None, palette=palettes)
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                    textcoords='offset points')
    title = "Multi-target Sentiment vs Multi-target No-Sentiment"
    plt.title(title)
    ax.set(ylabel='Macro average recall')
    title = title.replace(" ", "_")
    plt.savefig(output_path + title + ".png", dpi=100)
    plt.close()


def report_multi_target_crypto_oriented_sentiment_vs_no_sentiment(input_path_single_sentiment,
                                                                  input_path_single_no_sentiment, path_baseline,
                                                                  output_path, types,
                                                                  cryptocurrencies, percent):
    output_path = output_path + "multi_target_crypto_oriented/"
    i = 0
    report = i
    while i < len(cryptocurrencies):
        save_baseline_single_target = True
        df_report = pd.DataFrame()
        for k in types:
            path_multi_target_no_sentiment = "../modelling/techniques/forecasting/outputs_multi_" + str(
                percent) + "/outputs_multi_no_sentiment_" + str(percent) + "/" + k + "/multi_target/"
            path_multi_target_sentiment = "../modelling/techniques/forecasting/outputs_multi_" + str(
                percent) + "/outputs_multi_sentiment_" + str(percent) + "/" + k + "/multi_target/"
            for cluster in os.listdir(os.path.join(path_multi_target_no_sentiment, "clusters/")):
                for crypto in os.listdir(os.path.join(path_multi_target_no_sentiment, "clusters", cluster, "result")):
                    if crypto in cryptocurrencies[i]:
                        report = str(i)

                        if save_baseline_single_target:
                            file = open(path_baseline + crypto + "_macro_avg_recall.txt", "r")
                            macro_avg_recall_baseline = file.read()
                            df_report = df_report.append(
                                {'crypto': crypto, 'model type': "baseline",
                                 'macro_avg_recall': float(macro_avg_recall_baseline),
                                 'config': "standard"},
                                ignore_index=True)
                            file.close()

                            # find best configuration no sentiment
                            max_macro_avg_recall_no_sentiment = -1
                            config_no_sentiment = ""
                            for configuration in os.listdir(os.path.join(input_path_single_no_sentiment, crypto)):
                                df = pd.read_csv(
                                    os.path.join(input_path_single_no_sentiment, crypto, configuration,
                                                 "stats/macro_avg_recall.csv"),
                                    header=0)
                                if df["macro_avg_recall"][0] > max_macro_avg_recall_no_sentiment:
                                    max_macro_avg_recall_no_sentiment = df["macro_avg_recall"][0]
                                    config_no_sentiment = configuration
                            df_report = df_report.append(
                                {'crypto': crypto, 'model type': 'single target no sentiment',
                                 'macro_avg_recall': float(max_macro_avg_recall_no_sentiment),
                                 'config': config_no_sentiment}, ignore_index=True)

                            # find best configuration sentiment
                            max_macro_avg_recall_sentiment = -1
                            config_sentiment = ""
                            for configuration in os.listdir(os.path.join(input_path_single_sentiment, crypto)):
                                df = pd.read_csv(
                                    os.path.join(input_path_single_sentiment, crypto, configuration,
                                                 "stats/macro_avg_recall.csv"),
                                    header=0)
                                if df["macro_avg_recall"][0] > max_macro_avg_recall_sentiment:
                                    max_macro_avg_recall_sentiment = df["macro_avg_recall"][0]
                                    config_sentiment = configuration
                            df_report = df_report.append(
                                {'crypto': crypto, 'model type': 'single target sentiment',
                                 'macro_avg_recall': float(max_macro_avg_recall_sentiment),
                                 'config': config_sentiment},
                                ignore_index=True)

                        highest_macro_avg_recall = -1
                        best_conf = ""
                        for configuration in os.listdir(
                                os.path.join(path_multi_target_no_sentiment, "clusters", cluster, "result", crypto)):
                            df = pd.read_csv(
                                os.path.join(path_multi_target_no_sentiment, "clusters", cluster, "result", crypto,
                                             configuration,
                                             "stats/macro_avg_recall.csv"))
                            value = df['macro_avg_recall'][0]
                            if value > highest_macro_avg_recall:
                                highest_macro_avg_recall = value
                                best_conf = configuration
                        df_report = df_report.append(
                            {'crypto': crypto, 'model type': k + ' no sentiment',
                             'macro_avg_recall': float(highest_macro_avg_recall),
                             'config': best_conf},
                            ignore_index=True)
            for cluster in os.listdir(os.path.join(path_multi_target_sentiment, "clusters/")):
                for crypto in os.listdir(os.path.join(path_multi_target_sentiment, "clusters", cluster, "result")):
                    if crypto in cryptocurrencies[i]:
                        report = str(i)
                        highest_macro_avg_recall = -1
                        best_conf = ""
                        for configuration in os.listdir(
                                os.path.join(path_multi_target_sentiment, "clusters", cluster, "result", crypto)):
                            df = pd.read_csv(
                                os.path.join(path_multi_target_sentiment, "clusters", cluster, "result", crypto,
                                             configuration,
                                             "stats/macro_avg_recall.csv"))
                            value = df['macro_avg_recall'][0]
                            if value > highest_macro_avg_recall:
                                highest_macro_avg_recall = value
                                best_conf = configuration
                        df_report = df_report.append(
                            {'crypto': crypto, 'model type': k + ' sentiment',
                             'macro_avg_recall': float(highest_macro_avg_recall),
                             'config': best_conf},
                            ignore_index=True)
            save_baseline_single_target = False
        i += 1
        folder_creator(output_path, 0)
        df_report.to_csv(os.path.join(output_path, "multi_target_crypto_oriented_" + report + ".csv"), index=False)
        report_multi_target_crypto_oriented_plot_sentiment_vs_no_sentiment(df_report, output_path, report)


def report_multi_target_crypto_oriented_plot_sentiment_vs_no_sentiment(df, output_path, report):
    title = "Multi-target crypto oriented results"
    fig = plt.figure(figsize=(20, 8))
    flatui = ["#ec7063", "#a569bd", "#5dade2", "#27ae60", "#f4d03f", "#f39c12", "#b2babb", "#138d75", "#cc66ff",
              "#4dff4d", "#59b300", "#0000b3", "#ff0000"]
    palettes = sns.color_palette(flatui)
    # sns.set(font_scale=0.65, style='white')
    ax = sns.barplot(x="crypto", y="macro_avg_recall", hue="model type", data=df, palette=palettes)
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
    # Put a legend to the right side
    # ax.legend(loc='center right', bbox_to_anchor=(1.16, 0.5), ncol=1)
    ax.legend(loc='center right', bbox_to_anchor=(1.16, 0.5), ncol=1)
    plt.title(title)
    title = title.replace(" ", "_")
    ax.set(xlabel='Cryptocurrencies', ylabel='Macro average recall')
    plt.savefig(output_path + "/" + title + "_" + report + ".png", dpi=100, bbox_inches='tight')
    plt.close(fig)

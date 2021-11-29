import matplotlib.pyplot as plt
import numpy as np


# for missing values
def missing_values_by_year(df, PATH_TO_SAVE):
    # set the x-axis labels
    x_axis_values = df.columns.values
    for crypto in df.index:
        fig = plt.figure(figsize=(20, 10), dpi=70)
        ax = fig.add_subplot(111)
        y_axis_values = []
        for value in df.loc[crypto].values:
            y_axis_values.append(value)
        if np.max(y_axis_values) > 0:
            # insert data in the bar
            ax.bar(x_axis_values, y_axis_values)
            ax.set_title(crypto)
            ax.set_ylabel('Number of missing values')
            ax.set_xlabel('Year')
            # ax.set_xticklabels(x_axis_values)
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.grid(linewidth=0.2, color='black')
            plt.savefig(PATH_TO_SAVE + crypto + ".png")

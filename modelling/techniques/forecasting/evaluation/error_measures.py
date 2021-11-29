import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


def get_rmse(actual, prediction):
    return np.math.sqrt(mean_squared_error(actual, prediction))


# qui dimostro che l'accuracy score coincide con l'accuracy che viene riportato nel classification report.
# accuracy: somma di tutti i True positivi per ogni classe / somma di tutti gli elementi nella tabella, easy.

def get_accuracy(actual, prediction):
    # Constants
    """C = "Cat"
    F = "Fish"
    H = "Hen"
    # True values
    y_true = [C, C, C, C, C, C, F, F, F, F, F, F, F, F, F, F, H, H, H, H, H, H, H, H, H]
    # Predicted values
    y_pred = [C, C, C, C, H, F, C, C, C, C, C, C, H, H, F, F, C, C, C, H, H, H, H, H, H]

    print(accuracy_score(y_true,y_pred))
    # Print the confusion matrix
    print(metrics.confusion_matrix(y_true, y_pred))

    # Print the precision and recall, among other metrics
    print(metrics.classification_report(y_true, y_pred, digits=3))"""

    # print(jaccard_score(y_true,y_pred,average="macro"))
    # Print the confusion matrix
    """print(metrics.confusion_matrix(actual,prediction))

    # Print the precision and recall, among other metrics
    dict=metrics.classification_report(actual,prediction, digits=3,output_dict=True)
    print(metrics.classification_report(actual,prediction, digits=3))"""
    # print(dict.get('macro avg').get('precision'))
    return accuracy_score(actual, prediction)


# (True positive class 0/#elem_class_n + True positive class 1/#elem_class_1 +  True positive class 2/#elem_class_2) / number of classes =3
def get_classification_stats(actual, prediction):
    """true_positives=np.diagonal(confusion_matrix)
    #compute the number of element on each class
    partial_averages=[]
    i=0
    while i< len(true_positives):
        partial_averages.append(np.divide(true_positives[i],np.sum(confusion_matrix[i])))
        i+=1
    print(np.average(partial_averages))"""
    confusion_matrix = metrics.confusion_matrix(actual, prediction)
    print(confusion_matrix)
    confusion_matrix = pd.DataFrame(
        {'Stable': confusion_matrix[:, 0], 'Down': confusion_matrix[:, 1], 'Up': confusion_matrix[:, 2]})
    performances = metrics.classification_report(actual, prediction, digits=3, output_dict=True, zero_division=False)
    # print(metrics.classification_report(actual, prediction, digits=3))
    return confusion_matrix, performances


"""C = "Cat"
F = "Fish"
H = "Hen"
# True values
y_true = [C, C, C, C, C, C, F, F, F, F, F, F, F, F, F, F, H, H, H, H, H, H, H, H, H]
# Predicted values
y_pred = [C, C, C, C, H, F, C, C, C, C, C, C, H, H, F, F, C, C, C, H, H, H, H, H, H]
confusion_matrix=metrics.confusion_matrix(y_true,y_pred)
confusion_matrix = pd.DataFrame(
    {'Stable': confusion_matrix[:, 0], 'Down': confusion_matrix[:, 1], 'Up': confusion_matrix[:, 2]})

print(confusion_matrix)"""

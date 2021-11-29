import gc
import os
import random as rn
from itertools import product

import numpy as np
import pandas as pd
import tensorflow_core as tf_core
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow_core.python.keras.utils.np_utils import to_categorical

from modelling.techniques.forecasting.evaluation.error_measures import get_classification_stats
from modelling.techniques.forecasting.training.training import prepare_input_forecasting, fromtemporal_totensor, \
    get_training_validation_testing_set, train_single_target_model
from utility.folder_creator import folder_creator

np.random.seed(42)
rn.seed(42)
# stable results
tf_core.random.set_seed(42)

PREPROCESSED_PATH = "../preparation/preprocessed_dataset/cleaned/final/"


def save_results(macro_avg_recall_file, crypto_name, predictions_file, results_path):
    macro_avg_recall_file['symbol'].append(crypto_name)
    # accuracy
    confusion_matrix, performances = get_classification_stats(predictions_file['observed_class'],
                                                              predictions_file['predicted_class'])
    macro_avg_recall_file['macro_avg_recall'].append(performances.get('macro avg').get('recall'))

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
    pd.DataFrame(data=confusion_matrix).to_csv(results_path + 'confusion_matrix.csv', index=False)
    df_performances_1.to_csv(
        os.path.join(results_path, "performances_part1.csv"), index=False)
    pd.DataFrame(dict_perf_2).to_csv(
        os.path.join(results_path, "performances_part2.csv"), index=False)

    # serialization
    pd.DataFrame(data=predictions_file).to_csv(results_path + 'predictions.csv', index=False)
    pd.DataFrame(data=macro_avg_recall_file).to_csv(results_path + 'macro_avg_recall.csv', index=False)


def single_target(EXPERIMENT_PATH, DATA_PATH, TENSOR_DATA_PATH, window_sequences, list_num_neurons, learning_rate,
                  features_to_use, DROPOUT, EPOCHS, PATIENCE, BATCH_SIZE, test_set):
    #################### FOLDER SETUP ####################
    MODELS_PATH = "models"
    RESULT_PATH = "result"
    # starting from the testing set
    for crypto_name in os.listdir(DATA_PATH):
        # create a folder for data in tensor format
        folder_creator(TENSOR_DATA_PATH + "/" + crypto_name, 0)
        # create a folder for results
        folder_creator(EXPERIMENT_PATH + "/" + MODELS_PATH + "/" + crypto_name, 0)
        folder_creator(EXPERIMENT_PATH + "/" + RESULT_PATH + "/" + crypto_name, 0)
        for window, num_neurons in product(window_sequences, list_num_neurons):
            print('Current configuration: ')
            print("Crypto: ", crypto_name, "\t", "Window_sequence: ", window, "\t", "Neurons: ", num_neurons)
            predictions_file = {'symbol': [], 'date': [], 'observed_class': [], 'predicted_class': []}
            macro_avg_recall_file = {'symbol': [], 'macro_avg_recall': []}
            # New folders for this configuration
            configuration_name = "LSTM_" + str(num_neurons) + "_neurons_" + str(window) + "_days"
            # Create a folder to save
            # - best model checkpoint
            # - statistics (results)
            statistics = "stats"
            model_path = EXPERIMENT_PATH + "/" + MODELS_PATH + "/" + crypto_name + "/" + configuration_name + "/"
            results_path = EXPERIMENT_PATH + "/" + RESULT_PATH + "/" + crypto_name + "/" + configuration_name + "/" + statistics + "/"
            folder_creator(model_path, 0)
            folder_creator(results_path, 0)
            for date_to_predict in test_set:
                # format of dataset name: Crypto_DATE_TO_PREDICT.csv
                dataset_name = crypto_name + "_" + str(date_to_predict) + ".csv"
                dataset, features_without_date = \
                    prepare_input_forecasting(os.path.join(DATA_PATH, crypto_name), dataset_name, features_to_use)
                # print(dataset.dtypes)
                dataset_tensor_format = fromtemporal_totensor(np.array(dataset), window,
                                                              TENSOR_DATA_PATH + "/" + crypto_name + "/",
                                                              crypto_name + "_" + date_to_predict)

                # train, validation,test = get_training_validation_testing_set(dataset_tensor_format, date_to_predict)
                train, test = get_training_validation_testing_set(dataset_tensor_format, date_to_predict)
                #print('training set')
                #print(train)
                #print(train.shape)

                index_of_target_feature = features_without_date.index('trend')
                #print("INDEX")
                #print(index_of_target_feature)
                #remove the last day before the day to predict
                x_train = train[:, :-1, :index_of_target_feature]
                #print("X_TRAIN")
                #print(x_train)
                #print(x_train.shape)

                y_train = train[:, -1, index_of_target_feature]
                #print("Y_TRAIN")
                #print(y_train)
                #print(y_train.shape)

                x_test = test[:, :-1, :index_of_target_feature]
                '''print("X_TEST")
                print(x_test)
                print(x_test.shape)'''

                y_test = test[:, -1, index_of_target_feature]
                '''print("Y_TEST")
                print(y_test)
                print(y_test.shape)'''

                # change the data type, from object to float
                x_train = x_train.astype('float')
                x_test = x_test.astype('float')

                # one hot encode y

                #y_train = to_categorical(y_train - y_train.min())
                #y_test = to_categorical(y_test - y_test.min())
                y_train = to_categorical(y_train  + 1)
                y_test = to_categorical(y_test + 1)
                '''print(y_train)
                print(y_test)'''

                # batch size must be a factor of the number of training elements
                if BATCH_SIZE == None:
                    BATCH_SIZE = x_train.shape[0]

                model, history = train_single_target_model(x_train, y_train,
                                                           num_neurons=num_neurons,
                                                           learning_rate=learning_rate,
                                                           dropout=DROPOUT,
                                                           epochs=EPOCHS,
                                                           batch_size=BATCH_SIZE,
                                                           patience=PATIENCE,
                                                           num_categories=len(y_train[0]),
                                                           date_to_predict=date_to_predict,
                                                           model_path=model_path)
                # plot neural network's architecture
                plot_model(model, to_file=model_path + "neural_network.png", show_shapes=True,
                           show_layer_names=True, expand_nested=True, dpi=150)

                # plot loss
                """filename="model_train_val_loss_bs_"+str(BATCH_SIZE)+"_target_"+str(date_to_predict)
                plot_train_and_validation_loss(pd.Series(history.history['loss']),pd.Series(history.history['val_loss']),model_path,filename)

                #plot accuracy
                filename = "model_train_val_accuracy_bs_" + str(BATCH_SIZE) + "_target_" + str(date_to_predict)
                plot_train_and_validation_accuracy(pd.Series(history.history['accuracy']),
                                               pd.Series(history.history['val_accuracy']), model_path, filename)"""

                # Predict for each date in the validation set
                test_prediction = model.predict(x_test)
                # this is important!!
                K.clear_session()
                tf_core.random.set_seed(42)
                gc.collect()
                del model
                del dataset_tensor_format
                del dataset

                print("Num of entries for training: ", x_train.shape[0])
                # invert encoding: argmax of numpy takes the higher value in the array
                print("Predicting for: ", date_to_predict)
                print("Predicted: ", np.argmax(test_prediction))
                print("Actual: ", np.argmax(y_test))
                print("\n")

                # Saving the predictions on the dictionarie
                predictions_file['symbol'].append(crypto_name)
                predictions_file['date'].append(date_to_predict)
                predictions_file['observed_class'].append(np.argmax(y_test))
                predictions_file['predicted_class'].append(np.argmax(test_prediction))
            save_results(macro_avg_recall_file, crypto_name, predictions_file, results_path)
    return

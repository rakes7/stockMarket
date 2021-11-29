import os
import random as rn
import time
from itertools import product

import numpy as np
import tensorflow_core as tf_core

from modelling.techniques.forecasting.training.training import prepare_input_forecasting, fromtemporal_totensor, \
    get_training_validation_testing_set, train_single_target_model
from utility.folder_creator import folder_creator

np.random.seed(42)
rn.seed(42)
# stable results
tf_core.random.set_seed(42)

PREPROCESSED_PATH = "../preparation/preprocessed_dataset/cleaned/final/"


def single_target(EXPERIMENT_PATH, DATA_PATH, TENSOR_DATA_PATH, window_sequences, list_num_neurons, learning_rate,
                  testing_set, features_to_use, DROPOUT, EPOCHS, PATIENCE, number_of_days_to_predict, start_date,
                  end_date):
    #################### FOLDER SETUP ####################
    MODELS_PATH = "models"
    RESULT_PATH = "result"
    TIME_PATH = "time"

    for crypto in os.listdir(DATA_PATH):

        crypto_name = crypto.replace(".csv", "")
        # create a folder for data in tensor format

        folder_creator(TENSOR_DATA_PATH + "/" + crypto_name, 0)
        # create a folder for results
        folder_creator(EXPERIMENT_PATH + "/" + MODELS_PATH + "/" + crypto_name, 1)
        folder_creator(EXPERIMENT_PATH + "/" + RESULT_PATH + "/" + crypto_name, 1)

        # create folder for time spent
        folder_creator(EXPERIMENT_PATH + "/" + TIME_PATH + "/" + crypto_name, 1)

        dataset, features, features_without_date = \
            prepare_input_forecasting(PREPROCESSED_PATH, DATA_PATH, crypto_name, start_date, end_date, None,
                                      features_to_use)

        start_time = time.time()
        for window, num_neurons in product(window_sequences, list_num_neurons):
            print('Current configuration: ')
            print("Crypto_symbol: ", crypto, "\t", "Window_sequence: ", window, "\t", "Neurons: ", num_neurons)
            # print(np.array(dataset)[0]), takes the first row of the dataset (2018-01 2020...etc.)
            dataset_tensor_format = fromtemporal_totensor(np.array(dataset), window,
                                                          TENSOR_DATA_PATH + "/" + crypto_name + "/",
                                                          crypto_name)

            # DICTIONARY FOR STATISTICS
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
            folder_creator(model_path, 1)
            folder_creator(results_path, 1)

            accuracies = []
            # starting from the testing set
            for date_to_predict in testing_set:
                """the format of train and test is the following one:
                 [
                    [[Row1],[Row2]],
                    [[Row1],[Row2]],
                    ....
                    [[Row1],[Row2]],
                 ]
                thus for element accessing there are the following three indexes:
                  1)e.g [[items],[items]]
                  2)e.g [items],[items]
                  3)e.g items
                """
                # train, validation,test = get_training_validation_testing_set(dataset_tensor_format, date_to_predict)
                train, test = get_training_validation_testing_set(dataset_tensor_format, date_to_predict,
                                                                  number_of_days_to_predict)
                # ['2018-01-01' other numbers separated by comma],it removes the date.

                train = train[:, :, 1:]
                test = test[:, :, 1:]

                index_of_target_feature = features_without_date.index('Close')

                # print(index_of_target_feature)
                # remove the last day before the day to predict:
                # e.g date to predict 2019-01-07 thus the data about 2019-01-06 will be discarded.
                # e.g [[items],[items2],[items3]] becames [[items1],[items2]]
                # also, i will remove the "Close" feature, thanks to the third index (1)
                # x_train= train[:, :-1, index_of_target_feature:]
                # todo qua ho messo index of target feature
                # da rimettere
                # index_of_target_feature+=1
                # x_train = train[:, :-number_of_days_to_predict, :index_of_target_feature]
                x_train = train[:, :-number_of_days_to_predict, :]
                print("X_TRAIN")
                print(x_train)
                print(x_train.shape)

                # remove the last n-day before the days to predict, by doing -numberof
                # returns an array with all the values of the feature close
                # this contains values about the target feature!

                y_train = train[:, -number_of_days_to_predict:, index_of_target_feature]
                print("Y_TRAIN")
                print(y_train)
                print(y_train.shape)
                # print(y_train.shape)

                # x_val = validation[:, :-1, :]
                """print("X_VAL")
                print(x_val)
                print(x_val.shape)"""
                # remove the last day before the day to predict, by doing -1
                # returns an array with all the values of the feature close to predict!
                # y_val = validation[:, -1, index_of_target_feature]
                """print("Y_VAL")
                print(y_val)
                print(y_val.shape)"""
                # NOTE: in the testing set we must have the dates to evaluate the experiment without the date to forecast!!!
                # remove the day to predict
                # e.g date to predict 2019-01-07 thus the data about 2019-01-07 will be discarded.
                # e.g [[items],[items2],[items3]] becames [[items1],[items2]]
                # todo qua ho messo index of target feature
                # x_test = test[:, :-1, :]
                # x_test = test[:, :-number_of_days_to_predict, :index_of_target_feature]
                x_test = test[:, :-number_of_days_to_predict, :]
                print("X_TEST")
                print(x_test)
                # print(x_test.shape)
                # remove the last day before the day to predict, by doing -1
                # returns an array with all the values of the feature close to predict!
                y_test = test[:, -number_of_days_to_predict:, index_of_target_feature]
                print("Y_TEST")
                print(y_test)
                # print(y_test.shape)

                # change the data type, from object to float
                # print(x_train[0][0])
                x_train = x_train.astype('float')
                # print(x_train[0][0])
                y_train = y_train.astype('float')
                x_test = x_test.astype('float')
                y_test = y_test.astype('float')
                # print(y_test)
                # one hot encode y
                # y_train  = to_categorical(y_train)
                # y_test = to_categorical(y_test)
                # print(y_train)
                # print(y_test)
                # print(np.argmax(y_test))
                # batch size must be a factor of the number of training elements
                BATCH_SIZE = x_train.shape[0]
                # if the date to predict is the first date in the testing_set
                # if date_to_predict == testing_set[0]:
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
                """plot_model(model, to_file=model_path + "neural_network.png", show_shapes=True,
                           show_layer_names=True, expand_nested=True, dpi=150)

                #plot loss
                filename="model_train_val_loss_bs_"+str(BATCH_SIZE)+"_target_"+str(date_to_predict)
                plot_train_and_validation_loss(pd.Series(history.history['loss']),pd.Series(history.history['val_loss']),model_path,filename)

                #plot accuracy
                filename = "model_train_val_accuracy_bs_" + str(BATCH_SIZE) + "_target_" + str(date_to_predict)
                plot_train_and_validation_accuracy(pd.Series(history.history['accuracy']),
                                               pd.Series(history.history['val_accuracy']), model_path, filename)

                # Predict for each date in the validation set
                test_prediction = model.predict(x_test)
                # this is important!!
                K.clear_session()
                tf_core.random.set_seed(42)

                # changing data types
                #test_prediction = float(test_prediction)
                #test_prediction=test_prediction.astype("float")

                print("Num of entries for training: ", x_train.shape[0])
                # print("Num of element for validation: ", x_test.shape[0])
                #print("Training until: ", pd.to_datetime(date_to_predict) - timedelta(days=3))

                days = []
                i = number_of_days_to_predict-1
                while i > 0:
                    d = pd.to_datetime(date_to_predict) - timedelta(days=i)
                    days.append(d)
                    i -= 1
                days.append(pd.to_datetime(date_to_predict))

                # invert encoding: argmax of numpy takes the higher value in the array

                i=0
                for d in days:
                    print("Predicting for: ", d)
                    print("Predicted: ", np.argmax(test_prediction[i]))
                    print("Actual: ", np.argmax(y_test[i]))
                    i+=1
                print("\n")

                #todo RMSE AND ACCURACY
                # saving the accuracy on these predictions
                # Saving the predictions on the dictionarie
                i = 0
                for d in days:
                    predictions_file['symbol'].append(crypto_name)
                    predictions_file['date'].append(d)
                    predictions_file['observed_class'].append(np.argmax(y_test[i]))
                    predictions_file['predicted_class'].append(np.argmax(test_prediction[i]))
                    i += 1

            # Saving the accuracy into the dictionaries
            macro_avg_recall_file['symbol'].append(crypto_name)

            # accuracy
            performances= get_classification_stats(predictions_file['observed_class'], predictions_file['predicted_class'])
            macro_avg_recall_file['macro_avg_recall'].append(performances.get('macro avg').get('recall'))

            # serialization
            pd.DataFrame(data=predictions_file).to_csv(results_path + 'predictions.csv', index=False)
            pd.DataFrame(data=macro_avg_recall_file).to_csv(results_path + 'macro_avg_recall.csv', index=False)
            #confusion_matrix.to_csv(results_path + 'confusion_matrix.csv', index=False)
            #pd.DataFrame(data=performances).to_csv(results_path + 'performances.csv', index=False)
        time_spent=time.time() - start_time
        f=open(EXPERIMENT_PATH + "/" + TIME_PATH + "/" + crypto_name+"/"+"time_spent.txt","w+")
        f.write(str(time_spent))
        f.close()"""
    return

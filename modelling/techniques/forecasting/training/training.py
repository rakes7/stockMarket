import os

import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight, compute_class_weight


def prepare_input_forecasting(DATA_PATH, crypto, features_to_use):
    df = pd.read_csv(os.path.join(DATA_PATH, crypto), usecols=features_to_use)
    # features_without_symbols = [feature for feature in df.columns if not feature.startswith("symbol")]
    features_without_date_and_symbols = [feature for feature in df.columns if
                                         feature != "Date" and not feature.startswith("symbol")]
    return df, features_without_date_and_symbols


def fromtemporal_totensor(dataset, window_considered, output_path, output_name):
    try:
        # pickling is also known as Serialization
        # The pickle module is not secure. Only unpickle data you trust.
        # load is for de-serialize
        # allow_pickle=True else: Object arrays cannot be loaded when allow_pickle=False
        file_path = output_path + "/crypto_TensorFormat_" + output_name + "_" + str(window_considered) + '.npy'
        lstm_tensor = np.load(file_path, allow_pickle=True)
        print('(LSTM Version found!)')
        return lstm_tensor
    except FileNotFoundError as e:
        print('Creating LSTM version..')
        # an array in this format: [ [[items],[items]], [[items],[items]],.....]
        # -num of rows: window_considered
        # -num of columns: "dataset.shape[1]"
        # 1 is the number of elements in
        lstm_tensor = np.zeros((1, window_considered, dataset.shape[1]))
        # for i between 0 to (num of elements in original array - window + 1)
        """easy explanation through example:
             i:0-701 (730-30+1)
             i=0; => from day 0 + 30 days 
             i=1 => from day 1 + 30 days 
          """
        for i in range(dataset.shape[0] - window_considered + 1):
            # note (i:i + window_considered) is the rows selection.
            element = dataset[i:i + window_considered, :].reshape(1, window_considered, dataset.shape[1])
            lstm_tensor = np.append(lstm_tensor, element, axis=0)  # axis 0 in order to appen on rows

        # serialization
        output_path += "/crypto_"
        name_tensor = 'TensorFormat_' + output_name + '_' + str(window_considered)
        # since the first element is zero I'll skip it:
        lstm_tensor = lstm_tensor[1:, :]
        np.save(str(output_path + name_tensor), lstm_tensor)
        print('LSTM version created!')
        return lstm_tensor


def get_training_validation_testing_set(dataset_tensor_format, date_to_predict):
    train = []
    test = []
    index_feature_date = 0
    for sample in dataset_tensor_format:
        # Candidate is a date: 2018-01-30, for example.
        # -1 is used in order to reverse the list.
        # takes the last date in the sample: 2017-01-09, 2017-01..., ... ,  2017-02-2019
        # since the last date is 2017-02-2019, then it is before the date to predict for example 2019-03-05, so this sample
        # will belong to the training set.
        candidate = sample[-1, index_feature_date]
        candidate = pd.to_datetime(candidate)

        # if the candidate date is equal to the date to predict then it will be in test set.
        # it happens just one time for each date to predict.
        # Test will be: [[items]] in which the items goes N(30,100,200) days before the date to predict.
        # d_validation = pd.to_datetime(date_to_predict) - timedelta(days=3)
        """days=[]
        i=number_of_days_to_predict-1
        while i>0:
            d = pd.to_datetime(date_to_predict) - timedelta(days=i)
            days.append(d)
            i-=1
        days.append(pd.to_datetime(date_to_predict))"""
        if candidate == pd.to_datetime(date_to_predict):
            # remove the "Data" information
            sample = sample[:, 1:].astype('float')
            test.append(sample)
        elif candidate < pd.to_datetime(date_to_predict):
            # remove the "Data" information
            sample = sample[:, 1:].astype('float')
            train.append(sample)
    # return np.array(train), np.array(validation),np.array(test)
    return np.array(train), np.array(test)


def train_single_target_model(x_train, y_train, num_neurons, learning_rate, dropout, epochs, batch_size, patience,
                              num_categories,
                              date_to_predict, model_path='', model=None):
    # note: it's an incremental way to get a final model.
    #
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, mode='min'),
        ModelCheckpoint(
            monitor='val_loss', save_best_only=True, mode='min',
            filepath=model_path + 'lstm_neur{}-do{}-ep{}-bs{}-target{}.h5'.format(
                num_neurons, dropout, epochs, batch_size, date_to_predict))
    ]
    if model is None:
        model = Sequential()
        # Add a LSTM layer with 128/256 internal units.
        # model.add(LSTM(units=num_neurons,return_sequences=True,input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(units=num_neurons, input_shape=(x_train.shape[1], x_train.shape[2])))
        # reduce the overfitting
        model.add(Dropout(dropout))
        model.add(Dense(units=num_neurons, activation='relu', name='ReLu'))
        model.add(Dense(units=num_categories, activation='softmax', name='softmax'))
        # optimizer
        adam = Adam(learning_rate=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    y_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    d_class_weights = dict(enumerate(class_weights))

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.002,
                        verbose=0, shuffle=False, callbacks=callbacks, class_weight=d_class_weights)

    return model, history


def train_multi_target_model(x_train, y_trains_encoded, num_neurons, learning_rate, dropout, epochs, batch_size,
                             patience, num_categories,
                             date_to_predict, model_path='', model=None):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, mode='min'),
        ModelCheckpoint(
            monitor='val_loss', save_best_only=True, mode='min',
            filepath=model_path + 'lstm_neur{}-do{}-ep{}-bs{}-target{}.h5'.format(
                num_neurons, dropout, epochs, batch_size, date_to_predict))
    ]
    # note: it's an incremental way to get a final model.
    #
    inputs_stm = Input(shape=(x_train.shape[1], x_train.shape[2]))
    lstm = LSTM(units=num_neurons)(inputs_stm)
    # reduce the overfitting
    """lstm=Dropout(dropout)(lstm)
    lstm = Dense(units=num_neurons, activation='relu')(lstm)"""

    cryptocurrencies = []
    losses = {}
    losses_weights = {}
    y_train_dict = {}
    loss = "categorical_crossentropy"
    loss_weight = 1.0
    i = 0
    while i < len(y_trains_encoded):
        losses['trend_' + str(i)] = loss
        losses_weights['trend_' + str(i)] = loss_weight
        y_train_dict['trend_' + str(i)] = y_trains_encoded[i]

        crypto_model = LSTM(units=num_neurons)(inputs_stm)
        # reduce the overfitting
        crypto_model = Dropout(dropout)(crypto_model)
        crypto_model = Dense(units=num_neurons, activation='relu', name="ReLu_" + str(i))(crypto_model)
        crypto_model = Dense(units=num_categories, activation='softmax', name='trend_' + str(i))(crypto_model)
        cryptocurrencies.append(crypto_model)

        """crypto_model = Dropout(dropout)(lstm)
        crypto_model = Dense(units=num_neurons, activation='relu', name="ReLu_" + str(i))(crypto_model)
        crypto_model = Dense(units=num_categories, activation='softmax', name='trend_' + str(i))(crypto_model)
        cryptocurrencies.append(crypto_model)"""
        i += 1

    model = Model(
        inputs=inputs_stm,
        outputs=cryptocurrencies,
        name="multitarget")

    # initialize the optimizer and compile the model
    adam = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam, loss=losses, loss_weights=losses_weights,
                  metrics=["accuracy"])

    d_class_weights = {}
    i = 0
    while i < len(y_trains_encoded):
        y_integers = np.argmax(y_trains_encoded[i], axis=1)
        class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        d_class_weights["trend_" + str(i)] = dict(enumerate(class_weights))
        i = i+1

    history = model.fit(x_train, y_train_dict,
                        epochs=epochs, validation_split=0.002, batch_size=batch_size,
                        verbose=0, shuffle=False, callbacks=callbacks, class_weight=d_class_weights)
    return model, history


"""def train_model_new(x_train, y_trains_encoded, num_neurons, learning_rate, dropout, epochs, batch_size,patience, num_categories,
                date_to_predict,model_path='', model=None):
    #note: it's an incremental way to get a final model.
    #
    print(x_train.shape)
    inputs = Input(shape=(x_train.shape[1], x_train.shape[2]),batch_size=x_train.shape[0])
    #trend_btc= Sequential()
    trend_btc= LSTM(units=num_neurons)(inputs)
    print(LSTM)
    # reduce the overfitting
    trend_btc= Dropout(dropout)(trend_btc)
    trend_btc= Dense(units=num_neurons, activation='relu')(trend_btc)
    trend_btc=Dense(units=num_categories)(trend_btc)
    trend_btc= Activation('softmax',name="trend_btc")(trend_btc)


    # trend_btc= Sequential()
    trend_eth = LSTM(units=num_neurons)(inputs)
    # reduce the overfitting
    trend_eth = Dropout(dropout)(trend_eth)
    trend_eth = Dense(units=num_neurons, activation='relu')(trend_eth)
    trend_eth= Dense(units=num_categories)(trend_eth)
    trend_eth = Activation('softmax', name="trend_eth")(trend_eth)

    model = Model(
        inputs=inputs,
        outputs=[trend_btc,trend_eth],
        name="multitarget")

    losses = {
        "trend_btc": "categorical_crossentropy",
        "trend_eth": "categorical_crossentropy",
    }
    loss_weights = {"trend_btc": 1.0, "trend_eth": 1.0}
    # initialize the optimizer and compile the model
    adam = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam, loss=losses,  loss_weights=loss_weights,
                  metrics=["accuracy"])
    plot_model(model, to_file="neural_network.png", show_shapes=True,
               show_layer_names=True, expand_nested=True, dpi=150)

    history=model.fit(x_train, {"trend_btc": y_trains_encoded[0], "trend_eth": y_trains_encoded[1]},
                      epochs=epochs,  validation_split = 0.02,
                     verbose=0, shuffle=False)

    return model, history"""

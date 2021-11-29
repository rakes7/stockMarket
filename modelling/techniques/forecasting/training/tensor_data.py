import os

import numpy as np

from modelling.techniques.forecasting.training.training import prepare_input_forecasting, fromtemporal_totensor


# just for multi-target!
def generate_tensor_data(path, TENSOR_PATH, temporal_sequence_considered, MultiFeaturesToExclude):
    # for path in DATA_PATHS:
    # series = os.listdir(path)
    # for s in series:
    #   crypto_name = s.replace(".csv", "")
    #  os.makedirs(TENSOR_PATH + "/" + crypto_name, exist_ok=True)
    os.makedirs(TENSOR_PATH + "/" + "horizontal", exist_ok=True)
    features_to_exclude_from_scaling = MultiFeaturesToExclude

    data_compliant, features, features_without_date, scaler = prepare_input_forecasting(
        path + "/horizontal.csv",
        features_to_exclude_from_scaling)

    for temporal in temporal_sequence_considered:
        # print(s, "\t", temporal)
        fromtemporal_totensor(np.array(data_compliant), temporal,
                              TENSOR_PATH + "/horizontal",
                              "horizontal")

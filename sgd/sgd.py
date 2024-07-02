import numpy as np
from sklearn.linear_model import SGDClassifier

from flwr.common import NDArrays

def get_model_params(model: SGDClassifier) -> NDArrays:
    # Returns the parameters of a sklearn SGDClassifier model.
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(model: SGDClassifier, params: NDArrays) -> SGDClassifier:
    # Sets the parameters of a sklearn SGDClassifier model.
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: SGDClassifier):
# Sets initial parameters as zeros Required since model params are uninitialized until model.fit is called.
# But server asks for initial parameters from clients at launch. 
# Refer sklearn.linear_model.SGDClassifier documentation for more information.
    n_classes = 2 
    n_features = 13  # Number of features in dataset
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((1,n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((1,))
import numpy as np
import flwr as fl
from sklearn.naive_bayes import GaussianNB
from flwr.common import NDArrays

def get_model_parameters(model):
    return [model.theta_, model.var_, model.class_prior_, model.class_count_]

def set_model_parameters(model, parameters):
    model.theta_, model.var_, model.class_prior_, model.class_count_ = parameters
    return model

def set_initial_params(model: GaussianNB):
# Sets initial parameters as zeros Required since model params are uninitialized until model.fit is called.
# But server asks for initial parameters from clients at launch. 
# Refer sklearn.naive_bayes.GaussianNB documentation for more information.
    n_classes = 2 
    n_features = 13  # Number of features in dataset
    model.classes_ = np.array([i for i in range(n_classes)])
    model.theta_ = np.zeros((n_classes, n_features))
    model.var_ = np.ones((n_classes, n_features))
    model.class_count_ = np.zeros(n_classes)
    model.class_prior_ = np.array([1.0 / n_classes for _ in range(n_classes)])
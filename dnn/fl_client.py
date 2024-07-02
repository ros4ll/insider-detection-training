import numpy as np
import torch
from torch.utils.data import DataLoader
from flwr.common import log
from logging import INFO

import pandas as pd
import flwr as fl
import os


import functions
import dnn

class FlwrClient(fl.client.NumPyClient):
    def __init__(self,id) -> None:
        self.id = id
        # Load data 
        self.X_train,self.X_test,self.y_train,self.y_test,self.n_features = functions.load_data(test_split=True)
        self.X_train = self.X_train.astype(float)
        self.X_test = self.X_test.astype(float)
        # Initiate model
        self.model = dnn.DNN().to(DEVICE)
        train, test, train_dataset, test_dataset, trainloader, testloader = dnn.prepare_data(self.X_train, self.y_train, self.X_test,self.y_test)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        # Update model params
        self.set_parameters(parameters)
        log(INFO, f"Training data at {self.id}")
        # Train model
        trainloader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        dnn.train(self.model, trainloader)
        # Return updated params after fitting
        return self.get_parameters(config), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        log(INFO, f"Evaluating model at {self.id}")
        # Make predictions based on new model params
        testloader = DataLoader(self.test_dataset)
        loss, predictions, y_test = dnn.test(self.model, testloader)
        predictions = np.array(predictions)
        y_test = np.array(y_test)
        # Evaluate using eval_metrics function
        acc, rec, prec, f1s, tn, fp, fn, tp, _  = functions.eval_metrics(predictions, y_test)
        fpr = fp /(fp+tn)
        return loss, len(testloader.dataset), {"accuracy": acc, "recall": rec, "precision": prec, "f1s": f1s, "fpr": fpr}
    
def main():
    server_address=os.getenv("SERVER_ADDR")
    site_id = "site-"+os.getenv("CLIENT_NUM")
    client = FlwrClient(id=site_id)
    server_history = fl.client.start_client(
        server_address=server_address, 
        client=client.to_client()
        )
    return server_history
   
if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    history = main()  
    #utils.results_graph()
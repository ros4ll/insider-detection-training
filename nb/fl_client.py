import flwr as fl
import warnings
from sklearn.naive_bayes import GaussianNB
from flwr.common.logger import log
from flwr.common import NDArrays, Scalar
from logging import INFO, DEBUG
from typing import Dict, Optional, Tuple

import flwr as fl
import nb
import functions
import os

# Definir el modelo y la lógica de entrenamiento en el cliente
class FlwrClient(fl.client.NumPyClient):
    def __init__(self,id) -> None:
        self.id = id
        self.random_state = 42
        # Load data
        self.X_train,self.X_test,self.y_train,self.y_test,self.n_features = functions.load_data(test_split=True)
        # Initiate model
        self.model = GaussianNB()
        nb.set_initial_params(self.model)

    def get_parameters(self,config): #type:ignore
        return nb.get_model_parameters(self.model)
    
    def set_parameters(self, parameters):
        return nb.set_model_parameters(self.model, parameters)
    
    def fit(self, parameters, config): #type: ignore
        self.set_parameters(parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log(INFO, f"Training model at {self.id}")
            self.model.fit(self.X_train, self.y_train)
        log(INFO,f"Training finished for round {config['server_round']}")
        
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, Scalar]]:  # type: ignore
        log(INFO, f"Evaluating model at {self.id}")
        self.set_parameters(parameters)
        predictions = self.model.predict(self.X_test)
        acc, rec, prec, f1s, tn, fp, fn, tp, loss  = functions.eval_metrics(predictions, self.y_test)
        fpr = fp /(fp+tn)
        num_examples=len(self.X_test)
        metrics = {"accuracy": acc, "recall": rec, "precision": prec, "f1score":f1s, "fpr": fpr}
        log(INFO,f"loss:{loss}, num_examples{num_examples}, metrics:{metrics}")
        return loss, num_examples, metrics

def main():
    server_address=os.getenv("SERVER_ADDR")
    datapath = os.getenv("FL_DATAPATH")+"data.csv"
    site_id = "site-"+os.getenv("CLIENT_NUM")
    client = FlwrClient(id=site_id)
    fl.client.start_client(server_address=server_address, 
                                     client=client.to_client()
                                     )
   
if __name__ == "__main__":
    main()  
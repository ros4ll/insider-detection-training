import warnings

from sklearn.linear_model import SGDClassifier
from flwr.common.logger import log
from flwr.common import Scalar
from logging import INFO
from typing import Dict,Tuple

import flwr as fl
import sgd
import functions
import os

class FlwrClient(fl.client.NumPyClient):
    def __init__(self,id) -> None:
        self.id = id
        self.random_state = 42
        # Load data
        self.X_train,self.X_test,self.y_train,self.y_test,self.n_features = functions.load_data(test_split=True)
        # Initiate model
        self.model = SGDClassifier(
                loss="modified_huber",
                penalty='elasticnet',
                fit_intercept=True,
                learning_rate='adaptive',
                eta0=1e-3 ,
                max_iter=2000,
                alpha=0.001,
                warm_start=True,
                random_state=42
            )
        sgd.set_initial_params(self.model)

    def get_parameters(self, config):  # type: ignore
        return sgd.get_model_params(self.model)

    def fit(self, parameters, config):  # type: ignore
        # Update model params
        sgd.set_model_params(self.model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Train model
            log(INFO, f"Training model at {self.id}")
            self.model.fit(self.X_train, self.y_train)
        log(INFO,f"Training finished for round {config['server_round']}")
        # Return updated params after fitting
        return sgd.get_model_params(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, Scalar]]:  # type: ignore
        log(INFO, f"Evaluating model at {self.id}")
        sgd.set_model_params(self.model, parameters)
        # Make predictions based on new model params
        predictions = self.model.predict(self.X_test)
        # Evaluate using eval_metrics function
        acc, rec, prec, f1s, tn, fp, fn, tp, loss  = functions.eval_metrics(predictions, self.y_test)
        fpr = fp /(fp+tn)
        num_examples=len(self.X_test)
        metrics = {"accuracy": acc, "recall": rec, "precision": prec, "f1score":f1s, "fpr": fpr}
        log(INFO,f"loss:{loss}, num_examples{num_examples}, metrics:{metrics}")
        return loss, num_examples, metrics

def main():
    server_address=os.getenv("SERVER_ADDR")
    site_id = "site-"+os.getenv("CLIENT_NUM")
    client = FlwrClient(id=site_id)
    fl.client.start_client(
        server_address=server_address, 
        client=client.to_client()
        )
   
if __name__ == "__main__":
    main()    

import sgd
import os
import functions

import pandas as pd
import flwr as fl

from flwr.common import NDArrays, Scalar, Metrics
from flwr.common.logger import log
from sklearn.linear_model import SGDClassifier
from typing import Dict, Optional, Tuple, List
from logging import INFO, DEBUG

NUM_ROUNDS = 12

def fit_round(server_round: int) -> Dict:
    return {"server_round": server_round}

def get_evaluate_fn():
    log(INFO, f"Loading data...")
    X_test, y_test,_ = functions.load_data(test_split=False)
    # The `evaluate` function will be called after every round
    def evaluate( server_round: int, parameters: NDArrays, config: Dict[str, Scalar] ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Update model with the latest parameters
        model = SGDClassifier(
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
        sgd.set_initial_params(model)
        sgd.set_model_params(model, parameters)
        predictions = model.predict(X_test)
        acc, rec, prec, f1s, tn, fp, fn, tp, loss = functions.eval_metrics(predictions,y_test)
        fpr = fp /(fp+tn)
        return loss, {"accuracy": acc, "recall": rec, "precision": prec, "f1score":f1s, "fpr": fpr }

    return evaluate

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    log(INFO, f"Aggregating metrics")
    # Multiply metrics of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    fprs = [num_examples * m["fpr"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and average metrics
    accuracy = sum(accuracies) / sum(examples)
    recall = sum(recalls)/sum(examples)
    precision = sum(precisions)/sum(examples)
    fpr = sum(fprs)/sum(examples)

    return {"accuracy": accuracy, "recall": recall, "precision": precision, "fpr": fpr }

def main():
    server_address=os.getenv("SERVER_ADDR")
    # FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(),
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_round,
    )
    log(INFO, f"Starting server...")
    server_history = fl.server.start_server(
        server_address=server_address, 
        strategy=strategy, 
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS)
    )
    return server_history

if __name__ == "__main__":
    server_history = main()
    functions.plot_global_metric(server_history,"centralized","accuracy")
    functions.plot_global_metric(server_history,"centralized","precision")
    functions.plot_global_metric(server_history,"centralized","recall")
    functions.plot_global_metric(server_history,"centralized","f1score")
    functions.plot_global_metric(server_history,"centralized","fpr")
    functions.plot_loss(server_history,"centralized")
    functions.save_final_results(server_history,"centralized")
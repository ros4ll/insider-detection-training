from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import os
import torch
import dnn
import functions
import flwr as fl
from flwr.common import Metrics, NDArrays, Scalar
from flwr.common.logger import log
from logging import INFO

NUM_ROUNDS = 10

def fit_round(server_round: int) -> Dict:
    return {"server_round": server_round}

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

def get_evaluate_fn():
    log(INFO, f"Loading data...")
    X_test, y_test,_ = functions.load_data(test_split=False)
    X_test = X_test.astype(float)
    train, test, train_dataset, test_dataset, trainloader, testloader = dnn.prepare_data(X_test,y_test,X_test,y_test)
    # The 'evaluate' function will be called after every round
    def evaluate( server_round: int, parameters: NDArrays, config ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Update model with the latest parameters
        model = dnn.DNN().to(DEVICE)
        model.set_weights(parameters)
        loss, predictions, y_test = dnn.test(model, testloader)
        predictions = np.array(predictions)
        y_test = np.array(y_test)
        acc, rec, prec, f1s, tn, fp, fn, tp, loss = functions.eval_metrics(predictions,y_test)
        fpr = fp /(fp+tn)
        return loss, {"accuracy": acc, "recall": rec, "precision": prec, "f1score":f1s, "fpr": fpr }

    return evaluate

def main():
    server_address=os.getenv("SERVER_ADDR")
    # FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(),
        evaluate_metrics_aggregation_fn=weighted_average
    )
    log(INFO, f"Starting server...")
    server_history = fl.server.start_server(
        server_address=server_address, 
        strategy=strategy, 
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS)
    )
    return server_history

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    server_history = main()
    functions.plot_global_metric(server_history,"centralized","accuracy")
    functions.plot_global_metric(server_history,"centralized","precision")
    functions.plot_global_metric(server_history,"centralized","recall")
    functions.plot_global_metric(server_history,"centralized","f1score")
    functions.plot_global_metric(server_history,"centralized","fpr")
    functions.plot_loss(server_history,"centralized")
    functions.save_final_results(server_history,"centralized")
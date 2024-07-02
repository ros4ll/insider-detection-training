from logging import INFO
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flwr.common import log
import pandas as pd
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class DNN(nn.Module):
    
    def __init__(self) -> None:
        # Define DNN model of 4 layers with 13-64-32-16-1 neurons
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(13, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.5) # Reduce overfitting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Rectified Linear Unit Activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc6(x)) # For binary classification
        return x
    # aka get_model_params
    def get_weights(self):
        return [val.cpu().numpy() for _, val in self.state_dict().items()]
    # aka set_model_params
    def set_weights(self, weights):
        params_dict = zip(self.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)
    
def train(model, trainloader):
    # Loss function pos_weight=torch.tensor([10.0]).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 2

    for epoch in range(num_epochs):
        for features, labels in trainloader:
            # Restart grads
            optimizer.zero_grad()
            # Forward pass
            outputs = model(features.to(DEVICE))
            # Loss calculation
            loss = criterion(outputs, labels.to(DEVICE))
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

def test(model, testloader):
    criterion = torch.nn.BCELoss()
    model.eval()
    total_loss =0.0 
    y_true = []
    y_pred = []
    threshold = 0.3 # Decrease threashold for positive class
    with torch.no_grad():
        for features, labels in testloader:
            outputs = model(features.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE)).item()
            total_loss += loss * len(features.to(DEVICE))
            # Transform to binary label
            predicted = (outputs > threshold).cpu().numpy().astype(int)
            labels = labels.cpu().numpy().astype(int)
            y_true.extend(labels)
            y_pred.extend(predicted)

    average_loss = total_loss / len(testloader.dataset)
    return average_loss, y_pred, y_true

def prepare_data(X_train, y_train, X_test, y_test):
    log(INFO, "Loading data ...")
         # Training data
    train = pd.concat([X_train, y_train], axis=1)
    train_features = train.iloc[:, :-1].values
    train_labels = train.iloc[:, -1].values.reshape(-1, 1)  
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_features, dtype=torch.float),
        torch.tensor(train_labels, dtype=torch.float),
    )
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        # Test data
    test = pd.concat([X_test, y_test], axis=1)
    test_features = test.iloc[:, :-1].values
    test_labels = test.iloc[:, -1].values.reshape(-1, 1)  
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(test_features, dtype=torch.float),
        torch.tensor(test_labels, dtype=torch.float),
    )
    testloader = DataLoader(test_dataset)
    
    return train, test, train_dataset, test_dataset, trainloader, testloader
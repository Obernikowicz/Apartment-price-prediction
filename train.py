import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data_from_file(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=True, dtype=str)

    return data

def prepare_data(data):
    apartment_size_column = data[:, 0].astype(int)

    district_column = np.array(pd.get_dummies(data[:, 1]))

    no_of_bedrooms_column = data[:, 2].astype(int)

    no_of_bathrooms_column = data[:, 3].astype(int)
    
    current_year = datetime.now().year
    apartment_age = np.abs(data[:, 4].astype(int) - current_year)
    
    underground_parking = data[:, 5].astype(int)

    features = np.column_stack((
        apartment_size_column,
        district_column,
        no_of_bedrooms_column,
        no_of_bathrooms_column,
        apartment_age,
        underground_parking
    ))

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    prices = data[:, 6].astype(int)

    return (features, prices)

def train():
    train_dataset_path = 'mieszkania.csv'

    data = load_data_from_file(train_dataset_path)

    (X, Y) = prepare_data(data)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    Y = Y.view(Y.shape[0], 1)

    input_dim = X.shape[1]

    learning_rate = 0.01
    num_epochs = 10000
    model = nn.Linear(input_dim, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        Y_predicted = model(X)
        loss = criterion(Y_predicted, Y)
        
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        if epoch % 10 == 0:
            print(f'epoch {epoch + 1}: loss = {loss}')

train()

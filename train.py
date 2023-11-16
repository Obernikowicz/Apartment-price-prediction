import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml

class ApartmentDataset(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(data[:, :-1], dtype=torch.float32)
        self.prices = torch.tensor(data[:, -1], dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, id):
        return self.features[id], self.prices[id]
    
class ApartmentDataModule(LightningDataModule):
    def __init__(self, dataset_path, test_dataset_size, batch_size):
        super(ApartmentDataModule, self).__init__()
        self.dataset_path = dataset_path
        self.test_dataset_size = test_dataset_size
        self.batch_size = batch_size
    
    def prepare_data(self):
        data = np.genfromtxt(self.dataset_path, delimiter=',', skip_header=True, dtype=str)

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

        self.apartment_data = train_test_split(
            features, prices, 
            test_size=self.test_dataset_size
        )

    def setup(self, stage=None):
        (X_train, X_test, Y_train, Y_test) = self.apartment_data

        self.train_dataset = ApartmentDataset(np.column_stack((X_train, Y_train)))
        self.test_dataset = ApartmentDataset(np.column_stack((X_test, Y_test)))
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset)
    
    def get_input_dim(self):
        return self.apartment_data[0].shape[1]

class ApartmentLinearRegressionModel(LightningModule):
    def __init__(self, inpput_dim, criterion, learning_rate):
        super(ApartmentLinearRegressionModel, self).__init__()
        self.model = nn.Linear(inpput_dim, 1)
        self.criterion = criterion 
        self.learning_rate = learning_rate
    
    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_id):
        X, Y = batch
        Y_predicted = self(X)
        loss = self.criterion(Y_predicted, Y)
        return loss
    
    def test_step(self, batch, batch_id):
        X, Y = batch
        Y_predicted = self(X)
        loss = self.criterion(Y_predicted, Y)
        self.log('test_loss', loss)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr = self.learning_rate)

    def save_model(self, filepath='model.pth'):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath='model.pth'):
        self.load_state_dict(torch.load(filepath))

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    dataset_path = config['dataset_path']
    test_dataset_size = float(config['test_dataset_size'])
    learning_rate = float(config['learning_rate'])
    epochs = int(config['epochs'])
    batch_size = int(config['batch_size'])

    return (dataset_path,
            test_dataset_size,
            learning_rate,
            epochs,
            batch_size)

def run():
    config_path = 'config.yaml'

    (dataset_path,
    test_dataset_size,
    learning_rate,
    epochs,
    batch_size) = load_config(config_path)

    data_module = ApartmentDataModule(dataset_path=dataset_path, test_dataset_size=test_dataset_size, batch_size=batch_size)
    data_module.prepare_data()

    input_dim = data_module.get_input_dim()

    model = ApartmentLinearRegressionModel(input_dim, nn.MSELoss(), learning_rate)

    trainer = Trainer(max_epochs=epochs)

    trainer.fit(model, datamodule=data_module)

    model.save_model()

    loaded_model = ApartmentLinearRegressionModel(input_dim, nn.MSELoss(), learning_rate)
    loaded_model.load_model()

    test_dataloader = data_module.test_dataloader()
    trainer.test(loaded_model, dataloaders=test_dataloader)

run()

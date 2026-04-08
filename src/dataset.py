import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader


def load_dataset(dataset_dir='../data', window_size = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current_dir = os.path.dirname(os.path.abspath(__file__)) # location of the current file
    train_file = os.path.join(current_dir, dataset_dir, 'electricity_train.csv')
    test_file = os.path.join(current_dir, dataset_dir, 'electricity_test.csv')
    train_dataframe = pd.read_csv(train_file)
    test_dataframe = pd.read_csv(test_file)
    print("Dataframe sample data:")
    print(train_dataframe.head())
    print(f"Column names: {train_dataframe.columns}. Dataframe shape: {train_dataframe.shape}. Data types: {train_dataframe.dtypes}")

    # transofrms the timestamp column to datetime
    train_dataframe["timestamp"] = pd.to_datetime(train_dataframe["timestamp"])
    test_dataframe["timestamp"] = pd.to_datetime(test_dataframe["timestamp"])
    train_dataframe = train_dataframe.sort_values(by = "timestamp").reset_index(drop = True) # sort changes the row order, but it does not rebuild the index
    test_dataframe = test_dataframe.sort_values(by = "timestamp").reset_index(drop = True)

    train_values = np.array(train_dataframe["consumption"], dtype = np.float32)
    test_values = np.array(test_dataframe["consumption"], dtype = np.float32)

    def create_window(series, window_size) -> tuple[np.ndarray, np.ndarray]:
        X = []
        y = []
        for i in range(len(series) - window_size):
            X.append(series[i:i+window_size])
            y.append(series[i+window_size])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        
    X_train, y_train = create_window(train_values, window_size=window_size)
    X_test, y_test = create_window(test_values, window_size=window_size)

    return X_train, y_train, X_test, y_test

def get_dataset(window_size) -> tuple[Dataset, Dataset]:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    X_train, y_train, X_test, y_test = load_dataset(window_size=window_size)
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    return train_dataset, test_dataset

def get_data_loaders(train_dataset, test_dataset, batch_size = 64) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
    


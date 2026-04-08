from src.utils import get_device
from src.dataset import (get_dataset, get_data_loaders)
from src.model import ForecastModel
from src.train import train_model
import torch.nn as nn
import torch.optim as optim

def main():
    print("Q4:")
    device = get_device()
    window_size = 10
    train_dataset, test_dataset = get_dataset(window_size=window_size)
    train_loader, test_loader = get_data_loaders(train_dataset=train_dataset, test_dataset=test_dataset)
    model = ForecastModel(in_features=window_size)
    print(f"Model architecture: {model}")
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_model(model=model, train_loader=train_loader, val_loader=None,criterion=criterion, optimizer=optimizer, device=device, epochs=1)

    
if __name__ == "__main__":
    main()
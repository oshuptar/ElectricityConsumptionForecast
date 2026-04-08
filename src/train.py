import torch.nn as nn
import torch.optim as optim
import torch

def train_model(model : nn.Module, train_loader, test_loader, criterion, optimizer, device, epochs):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X).squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        mae = evaluate_model(model, test_loader, mae_criterion, device)
        rmse = evaluate_model(model, test_loader, rmse_criterion, device)
        mape = evaluate_model(model, test_loader, mape_criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss / len(train_loader):.4f}")
        print("Test results:")
        print(f"Mean Absolute Error {mae}\nRoot Mean Squared Error {rmse}\nMean Absolute percentage error {mape}%\n")

    return model

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X).squeeze(-1)
            loss = criterion(pred, y)
            total_loss += loss.item()

    return total_loss / len(dataloader)
        
def mae_criterion(pred : torch.tensor, value: torch.tensor):
    return torch.mean(torch.abs(pred - value))

def rmse_criterion(pred: torch.tensor, value: torch.tensor):
    return torch.sqrt(torch.mean((pred - value) ** 2))

def mape_criterion(pred: torch.tensor, value: torch.tensor):
    eps = 1e-8
    percentage_errors = torch.abs((pred - value) / torch.clamp(torch.abs(value), min=eps))
    return torch.mean(percentage_errors) * 100 # may become unstable since values in the dataset are normalizes and close to zero

import torch.nn as nn
import torch.optim as optim
import torch

def train_model(model : nn.Module, train_loader, val_loader, criterion, optimizer, device, epochs):
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
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss / len(train_loader):.4f}")

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
        

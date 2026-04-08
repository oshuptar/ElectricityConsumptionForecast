import torch
import numpy as np
import pandas as pd
import torch.nn as nn

def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU")
    return device


def build_comparison_dataframe(model, test_loader, test_df, window_size, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X).squeeze(-1)
            predictions.append(pred.cpu().numpy())
            actuals.append(y.cpu().numpy())
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    timestamps = test_df["timestamp"].iloc[window_size:].reset_index(drop=True)
    results_df = pd.DataFrame({
        "timestamp": timestamps,
        "actual": actuals,
        "predicted": predictions
    })

    return results_df

def forecast(model : nn.Module, initial_window, steps: int, device):
    model.eval()
    model.to(device)
    forecast = []
    current_window = np.array(initial_window, dtype=np.float32).copy()
    with torch.no_grad():
        for _ in range(steps):
            x = torch.from_numpy(current_window.astype(np.float32)).unsqueeze(0).to(device)
            out = model(x)
            prediction = out.squeeze().cpu().item()
            forecast.append(prediction)
            current_window = np.concatenate([current_window[1:], [prediction]])

    return np.array(forecast, dtype=np.float32)

def build_forecast_dataframe(forecast, start_date = pd.to_datetime("2015-01-01")):
    timestamps = pd.date_range(start=start_date, periods=len(forecast), freq="15min")
    return pd.DataFrame({
        "timestamp": timestamps,
        "consumption": forecast
    })

def get_hidden_layer1_size():
    return [64, 128]

def get_hidden_layer2_size():
    return [32, 64]

def get_learning_rate():
    return [1e-3, 3e-4]

def get_batch_size():
    return [256, 512]
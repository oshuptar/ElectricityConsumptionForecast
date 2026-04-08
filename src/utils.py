import torch

def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU")
    return device

def get_hidden_layer1_size():
    return [64, 128]

def get_hidden_layer2_size():
    return [32, 64]

def get_learning_rate():
    return [1e-3, 3e-4]

def get_batch_size():
    return [256, 512]
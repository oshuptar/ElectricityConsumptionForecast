from torch import nn


class ForecastModel(nn.Module):
    # takes in_features, which is equal to the window size and out_features which is a next prediction value
    def __init__(self, in_features, out_features = 1):
        
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features)
        )
    
    def forward(self, x):
        return self.net(x)

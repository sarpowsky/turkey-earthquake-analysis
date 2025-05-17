
import torch
import torch.nn as nn

class EarthquakeNN(nn.Module):
    def __init__(self, input_size):
        super(EarthquakeNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# Function to load model
def load_model(model_path, input_size, device='cpu'):
    model = EarthquakeNN(input_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

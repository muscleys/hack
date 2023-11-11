from flask import Flask, render_template, request
import rasterio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))]) # Only one value because gray-scale

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding_mode='zeros', device=device)  # Increased filters
        self.bn1 = nn.BatchNorm2d(16, device=device)  # Batch normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.dp1 = nn.Dropout2d(0.25)  # Dropout

        self.conv2 = nn.Conv2d(16, 32, 5, device=device)  # Increased filters
        self.bn2 = nn.BatchNorm2d(32, device=device)  # Batch normalization

        self.fc1 = nn.Linear(5408, 120, device=device)  # Adjusted input size
        self.dp2 = nn.Dropout(0.5)  # Dropout after fully connected
        self.fc2 = nn.Linear(120, 48, device=device)
        self.fc3 = nn.Linear(48, 5, device=device)
        self.fc4 = nn.Linear(5, 2, device=device)

    def _up_to_features(self, x):
        # Conv1 + BatchNorm + Activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dp1(x)

        # Conv2 + BatchNorm + Activation
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dp1(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Linear links in between + Dropout + Activation
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x

    def forward(self, x):
        x = self._up_to_features(x)
        x = F.relu(x)

        x = self.fc4(x)
        return x

# Load your PyTorch model
model = Net()
model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu'))) # Set the model to evaluation mode

def predict(image_whatever):
    img = rasterio.open_somehow(image_whatever)
    image_array = img.read(1)
    image = np.asarray(image_array)
    image = transform(image)
    
    prediction = model(image)


    return image

    
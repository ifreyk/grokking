#%%
import torch
import numpy as np
import pandas as pd
import sklearn as sc
from sklearn.datasets import make_regression
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=1000, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.FloatTensor(y_train.reshape(-1, 1)).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.FloatTensor(y_test.reshape(-1, 1)).to(device)
#%%
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.layer1 = nn.Linear(1000, 100)
        self.batch_norm1 = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(0.2)
        
        self.layer2 = nn.Linear(100, 100)
        self.batch_norm2 = nn.BatchNorm1d(100)
        self.dropout2 = nn.Dropout(0.2)
        
        self.layer3 = nn.Linear(100, 100)
        self.batch_norm3 = nn.BatchNorm1d(100)
        self.dropout3 = nn.Dropout(0.2)

        self.layer4 = nn.Linear(100, 100)
        self.batch_norm4 = nn.BatchNorm1d(100)
        self.dropout4 = nn.Dropout(0.2)
        
        self.output_layer = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        
        x = torch.relu(self.layer2(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        
        x = torch.relu(self.layer3(x))
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        
        x = self.output_layer(x)
        return x
#%%
# Instantiate the model
model = RegressionModel().to(device)
weight_decay = 1e-4
# Print the model summary to see the total number of parameters
print(model)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=weight_decay)
#%%
# Train the model
num_epochs = 1000000
train_loss_list = []
test_loss_list = []
weight_norm_list = []
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    train_loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    weight_norm_all = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight_norm = torch.norm(param, p=2).item()
                weight_norm_all.append(weight_norm)
    weight_norm_mean = np.mean(weight_norm_all)
    weight_norm_list.append(weight_norm_mean)
    # Print progress
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss.item():.4f}, Weight norm: {weight_norm_mean}')
    train_loss_list.append(train_loss)
    with torch.no_grad():
        model.eval()
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_loss_list.append(test_loss)
        print(f'Test Loss: {test_loss.item():.4f}')

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
from time import process_time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#%%
def testgpu():
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
    t0 = process_time()
    x = torch.ones(n1, device=mps_device)
    y = x + torch.rand(n1, device=mps_device)
    t1 = process_time()
    print(f"Total time with gpu ({n1}): {t1-t0}")
    t0 = process_time()
    x = torch.ones(n2, device=mps_device)
    y = x + torch.rand(n2, device=mps_device)
    t1 = process_time()
    print(f"Total time with gpu ({n2}): {t1-t0}")

def testcpu():
    t0 = process_time()
    x = torch.ones(n1)
    y = x + torch.rand(n1)
    t1 = process_time()
    print(f"Total time with cpu ({n1}): {t1-t0}")
    t0 = process_time()
    x = torch.ones(n2)
    y = x + torch.rand(n2)
    t1 = process_time()
    print(f"Total time with cpu ({n2}): {t1-t0}")
#%%
n1 = 10000
n2 = 100000000
testcpu()
testgpu()
#%%
data = pd.read_pickle('data_joined.pkl').T
annotation = pd.read_csv('descriptions_samples.csv',index_col=0)
y = annotation['age']
cg_order = data.iloc[:, :].corrwith(y).abs().sort_values(ascending=False)
#%%
top = 1000
top_sites = cg_order.index[:top]
#%%
X, y = data[top_sites].to_numpy(), y.to_numpy()
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
    def __init__(self,input_dim=1000,intermediate_dim=800, dropout=0.1):
        super(RegressionModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, intermediate_dim)
        self.batch_norm1 = nn.BatchNorm1d(intermediate_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.layer2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.batch_norm2 = nn.BatchNorm1d(intermediate_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        self.layer3 = nn.Linear(intermediate_dim, intermediate_dim)
        self.batch_norm3 = nn.BatchNorm1d(intermediate_dim)
        self.dropout3 = nn.Dropout(dropout)

        self.layer4 = nn.Linear(intermediate_dim, intermediate_dim)
        self.batch_norm4 = nn.BatchNorm1d(intermediate_dim)
        self.dropout4 = nn.Dropout(dropout)
        
        self.output_layer = nn.Linear(intermediate_dim, 1)
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

        x = torch.relu(self.layer4(x))
        x = self.batch_norm4(x)
        x = self.dropout4(x)
        
        x = self.output_layer(x)
        return x
#%%
model = RegressionModel(input_dim=top, intermediate_dim=800, dropout=0.1).to(device)
weight_decay = 1e-3
lr = 0.001
# Print the model summary to see the total number of parameters
print(model)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=1.)
#%%
# Train the model
num_epochs = 1000000
train_loss_list = []
test_loss_list = []
weight_norm_list = []
df = pd.DataFrame(index=range(0,num_epochs))
#%%
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    train_loss = criterion(outputs, y_train)
    df.loc[epoch,'learning_rate'] = optimizer.param_groups[-1]['lr']
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
    df.loc[epoch,'weight_norm_mean'] = weight_norm_mean
    df.loc[epoch,'train_loss'] = train_loss.item()
    weight_norm_list.append(weight_norm_mean)
    # Print progress
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss.item():.4f}, Weight norm: {weight_norm_mean}')
    #train_loss_list.append(train_loss)
    with torch.no_grad():
        model.eval()
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        #test_loss_list.append(test_loss)
        print(f'Test Loss: {test_loss.item():.4f}')
        df.loc[epoch,'test_loss'] = test_loss.item()
    if epoch%10000==0:
        df.to_csv('result.csv',index=False)
        torch.save(model.state_dict(), 'model')
#%%
import matplotlib.pyplot as plt
plot_df = df[~df['test_loss'].isna()]
plt.plot(df.index, df['test_loss'], marker='o')
plt.xlabel('Epoch')
plt.ylabel('Total Weight Norm')
plt.title('Total Weight Norm vs. Epoch')
plt.grid(True)
plt.show()
#%%
plot_df = df[~df['test_loss'].isna()]
plt.plot(df.index, df['test_loss'], marker='o')
plt.xlabel('Epoch')
plt.xscale('log')
plt.ylabel('Test loss')
plt.ylim([0,1000])
plt.title('Test loss vs. Epoch')
plt.yticks(range(0, 1000, 50))
plt.grid(True)
plt.show()
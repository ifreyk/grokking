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
#%%
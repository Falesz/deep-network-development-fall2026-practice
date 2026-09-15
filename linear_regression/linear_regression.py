import pandas as pd
import torch
import torch.nn as nn

# Load the dataset
data = pd.read_csv("linear_data.csv", names = ["value1", ["value2"])

# Create pytorch tensors
x = torch.tensor(data["value1"].values, dtype = torch.float32).view(-1, 1)
y = torch.tensor(data["value2"].values, dtype = torch.float32).view(-1, 1)

# Create a model

# Define loss function and optimizer

# Train the model

# Test the model

# Persist the model

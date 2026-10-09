import pandas as pd
import torch
import torch.nn as nn
import os

# Load the dataset
data = pd.read_csv("linear_data.csv", names = ["x", "y"]) # x is the input feature, y is the true output target

# Create pytorch tensors
x = torch.tensor(data["x"].values, dtype = torch.float32).view(-1, 1)
y = torch.tensor(data["y"].values, dtype = torch.float32).view(-1, 1)

# Create a model
# Since we have an almost perfect line of data points along the y = 3x + 10 line, a single neuron capable of learning weight and bias parameter is sufficient
model = nn.Linear(1, 1)

# Training hyperparameters
num_of_epochs = 50_000 # Training will run for this many epochs by default
min_num_of_epochs = 50 # Training will run for this many epochs irregardless of recent loss values
learning_rate = 0.001

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# History and early stopping condition
weight_history = []
bias_history = []
early_stopping_condition = lambda : abs(weight_history[-1] - weight_history[-2]) < 0.00005 and abs(bias_history[-1] - bias_history[-2]) < 0.00005

# Train the model
print("Starting training the model...")
for epoch in range(num_of_epochs):
    prediction = model(x)

    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    weight_history.append(model.weight.item())
    bias_history.append(model.bias.item())
    if epoch >= min_num_of_epochs and early_stopping_condition():
        print(f"Early stopping condition hit at epoch {epoch}! Breaking out of training loop...")
        break

# Test the model
print("Predictions:\n")
with torch.no_grad():
    for value in [x for x in range(0, 101, 5)]: # integers between 0 and 100 with steps of 5
        prediction = model(torch.tensor([[float(value)]]))
        print(f"{value} -> {prediction.item():.4f}")

# Persist the model
print("Persisting the constructed model...")
os.makedirs("models", exist_ok = True)
torch.save(model.state_dict(), "models/linear_regression_model.pth")

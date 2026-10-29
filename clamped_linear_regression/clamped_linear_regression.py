import pandas as pd
import torch
import torch.nn as nn
import os

# Load the dataset
data = pd.read_csv("clamped_linear_data.csv", names = ["x", "y"]) # x is the input feature, y is the true output target

# Create pytorch tensors
x = torch.tensor(data["x"].values, dtype = torch.float32).view(-1, 1)
y = torch.tensor(data["y"].values, dtype = torch.float32).view(-1, 1)

# Create a model
class ClampedLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # The model's job is to learn y = min(w×x + b, C) where x is the input, C is a boundary value where y caps out and
        # becomes a horizontal line / plateu and w and b are the positive slopes and intercepts of a line onto which points 
        # can be fitted below C. This also means that there is a value x' which is the boundary point on the input side
        # beyond which outputs would become the constant C

        # In terms of Linear neurons and ReLU activations the above function can be expressed as y = C - w×ReLU(-x + x')

        # The idea for the input layer is that it should learn z = -1x + x'
        # The sign of z tells us whether the input is below (positive sign) or above (negative sign) the boundary x'
        # The value of z tells us how far away we are from the boundary x'
        self.input_layer = nn.Linear(1, 1)
        with torch.no_grad():
            self.input_layer.weight.fill_(-1.0) # We know from the plan that the weight here should be a constant -1
        self.input_layer.weight.requires_grad = False

        # The idea for the output layer is that it should learn y = -w×ReLU(z) + C
        # Since the "scale" of z is still in input units, it doesn't yet account for the slope of the true line
        # w is the transformation that converts our values from input units to output units
        # w will ideally be negative, as we only have the input distance from the boundary x' and build up an output value
        # in the opposite direction first, which will then be corrected by the constant value of C
        self.output_layer = nn.Linear(1, 1)
    
    def forward(self, x):
        z = self.input_layer(x)
        y = self.output_layer(torch.relu(z))
        return y
    
model = ClampedLinearRegressionModel()
    
# Training hyperparameters
num_of_epochs = 50_000 # Training will run for this many epochs by default
learning_rate = 0.001

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Train the model
print("Starting to train the model...")
for epoch in range(num_of_epochs):
    prediction = model(x)

    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Inspect the model
print("\nSample predictions:")
with torch.no_grad():
    for value in [x for x in range(0, 101, 5)]: # integers between 0 and 100 with steps of 5
        prediction = model(torch.tensor([[float(value)]]))
        print(f"{value} -> {prediction.item():.4f}")
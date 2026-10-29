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

        # The idea for the input layer is that it takes the x input feature from the dataset
        # and fits it's output z to the z = 3x + 10 line, giving us good output candidates for x values between 0 and 63.3333...
        # TODO: THIS IDEA IS FUNDAMENTALLY FLAWED: THE INPUT LAYER HAS NO CHANCE TO LEARN 3x + 10 BECAISE THE TRAINING DATA DOESNT
        # SUGGEST THAT AT ALL 
        self.input_layer = nn.Linear(1, 1) # outputs values called z

        # The idea for the shift layer is that it should learn u = 1z - 200. The sign of u tells us whether the right output
        # would be z itself or the constant 200. A negative u means that 3x + 10 was not above 200, and so that value itself
        # should be the output. A positive u means that 3x + 10 was above 200, meaning 200 should be the output.
        # Passing this through a ReLU activation gives us the overhead above 200.
        # For example for u = -30 a ReLU activation gives us 0, telling us that z didn't go over the 200 upper boundary.
        # For example for u = 20 a ReLU activation gives us 20, telling us that z was above the 200 upper boundary by 20
        # Thus ReLU(u) can later be used to be subtracted from z, giving us back the value of z or 200, whichever is lesser 
        self.shift_layer = nn.Linear(1, 1)

        # The idea for the output layer is that it takes z and ReLU(u) as inputs and learns to calculate y = 1z - 1×ReLU(u) + 0
        # Simplified: y = z - ReLU(u)
        self.output_layer = nn.Linear(2, 1)
    
    def forward(self, x):
        # No calculation logic is hardcoded in here, the output is governed solely by the parameters the model learns
        # within the confines of the above architecture
        z = self.input_layer(x)
        u = self.shift_layer(z)
        y = self.output_layer(torch.cat((z, torch.relu(u)), dim = 1))
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
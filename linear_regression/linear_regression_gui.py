import torch
import torch.nn as nn
import tkinter as tk

# Recreate the model architecture from the actual linear regression script
model = nn.Linear(1, 1)

# Load the state dictionary
state_dict = torch.load("models/linear_regression_model.pth")

# Load the state dictionary into the reconstructed model
model.load_state_dict(state_dict)

# Put the model into inference mode so that it will satisfy actual requests for predictions from the user
model.eval()

# Create window for the user to interact with
window = tk.Tk()
window.title("Linear regression GUI")

input_label = tk.Label(window, text = "Enter a number between 0 and 100: ")
input_field = tk.Entry(window)

input_field.bind("<Return>", lambda event : output_prediction.config(
    text = f"{model(torch.tensor([float(input_field.get())])).item():.2f}"
))

output_label = tk.Label(window, text = "According to the linear regression model, the output should be: ")
output_prediction = tk.Label(window, text = f"NONE")

# Start up the window
input_label.pack()
input_field.pack()
output_label.pack()
output_prediction.pack()
window.mainloop()
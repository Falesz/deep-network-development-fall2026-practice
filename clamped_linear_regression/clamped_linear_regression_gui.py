import torch
import torch.nn as nn

import numpy as np

import tkinter as tk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from clamped_linear_regression import ClampedLinearRegressionModel

# Instantiate an untrained model
model = ClampedLinearRegressionModel()

# Load the state dictionary
state_dict = torch.load("models/clamped_linear_regression_model.pth")

# Load the state dictionary into the reconstructed model
model.load_state_dict(state_dict)

# Put the model into inference mode so that it will satisfy actual requests for predictions from the user
model.eval()

# Create window for the user to interact with
window = tk.Tk()
window.title("Clamped linear regression GUI")

input_label = tk.Label(window, text = "Enter a number between 0 and 100: ")
input_field = tk.Entry(window)

def handle_input():
    user_input = float(input_field.get())
    model_output = model(torch.tensor([user_input])).item()
    output_prediction.config(text = f"{model_output:.2f}")
    axes.plot(
        user_input, model_output, "ro",
        label = "Prediction" if "Prediction" not in axes.get_legend_handles_labels()[1] else "_nolegend_"
    )
    axes.annotate(
        f"({user_input}, {model_output:.2f})",
        (user_input, model_output),
        textcoords = "offset points",
        xytext = (10, 0),
        fontsize = 6
    )
    axes.legend()
    canvas.draw()

input_field.bind("<Return>", lambda event : handle_input())

output_label = tk.Label(window, text = "According to the clamped linear regression model, the output should be: ")
output_prediction = tk.Label(window, text = f"NONE")

# Plotting a figure with the true function
figure = Figure(figsize = (5, 4), dpi = 100)
axes = figure.add_subplot(1, 1, 1)

x_values = np.linspace(0, 100, 200)
y_values = np.minimum(3 * x_values + 10, 200)

axes.plot(x_values, y_values, label = "True function")
axes.set_title("Clamped linear regression: y = min(3x + 10, 200)")
axes.set_xlabel("Input (x)")
axes.set_ylabel("Output (y)")
axes.grid(True)
axes.legend()

canvas = FigureCanvasTkAgg(figure, master = window)
canvas.draw()

# Start up the window
input_label.pack()
input_field.pack()
output_label.pack()
output_prediction.pack()
canvas.get_tk_widget().pack()
window.mainloop()
import torch
import torch.nn as nn

import tkinter as tk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        fontsize = 8
    )
    axes.legend()
    canvas.draw()

input_field.bind("<Return>", lambda event : handle_input())

output_label = tk.Label(window, text = "According to the linear regression model, the output should be: ")
output_prediction = tk.Label(window, text = f"NONE")

# Plotting a figure with the true line
figure = Figure(figsize = (5, 4), dpi = 100)
axes = figure.add_subplot(1, 1, 1)

x_values = np.linspace(0, 100, 200)
y_values = 3 * x_values + 10

axes.plot(x_values, y_values, label = "True line")
axes.set_title("Linear regression: y = 3x + 10")
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
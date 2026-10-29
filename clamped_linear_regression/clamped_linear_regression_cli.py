import torch
import torch.nn as nn

from clamped_linear_regression import ClampedLinearRegressionModel

# Instantiate an untrained model
model = ClampedLinearRegressionModel()

# Load the state dictionary
state_dict = torch.load("models/clamped_linear_regression_model.pth")

# Load the state dictionary into the reconstructed model
model.load_state_dict(state_dict)

# Put the model into inference mode so that it will satisfy actual requests for predictions from the user
model.eval()

print("Clamped linear regression commandline interface")
print("Valid inputs are integers between 0 and 100. You may exit the program by inputting an empty string.")
print("When the model is trained on default generated data, the loaded model will try to approximate the function y = min(3x + 10, 200).")
print("This commandline interface has no way to detect whether the model has been trained on custom data.")

while True:
    raw_user_input_feature = input("Please enter an integer between 0 and 100: ")
    if not raw_user_input_feature:
        break

    user_input_feature = torch.tensor([float(raw_user_input_feature)])

    model_prediction = model(user_input_feature)
    print(f"The loaded model's prediction: {raw_user_input_feature} -> {model_prediction.item():.4f}")

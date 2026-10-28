import torch
import torch.nn as nn

# Recreate the model architecture from the actual linear regression script
model = nn.Linear(1, 1)

# Load the state dictionary
state_dict = torch.load("models/linear_regression_model.pth")

# Load the state dictionary into the reconstructed model
model.load_state_dict(state_dict)

# Put the model into inference mode so that it will satisfy actual requests for predictions from the user
model.eval()

print("Linear regression commandline interface")
print("Valid inputs are integers between 0 and 100. You may exit the program by inputting an empty string.")
print("The loaded model will try to approximate the line y = 3x + 10.")

while True:
    raw_user_input_feature = input("Please enter an integer between 0 and 100: ")
    if not raw_user_input_feature:
        break

    user_input_feature = torch.tensor([float(raw_user_input_feature)])

    model_prediction = model(user_input_feature)
    print(f"The loaded model's prediction: {raw_user_input_feature} -> {model_prediction.item():.4f}")

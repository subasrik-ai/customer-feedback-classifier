
import torch

# Step 1: Define input values
x1 = torch.tensor(1.0)
x2 = torch.tensor(0.0)

# Step 2: Define weights
w1 = torch.tensor(0.5)
w2 = torch.tensor(-0.3)

# Step 3: Define bias
bias = torch.tensor(0.1)

# Step 4: Calculate weighted sum
z = (x1 * w1) + (x2 * w2) + bias

# Step 5: Apply sigmoid activation function
output = torch.sigmoid(z)

# Step 6: Display results
print("Weighted sum:", z.item())
print("Neuron output:", output.item())


# Step 7: Build a complete neural network

# Two input values
inputs = torch.tensor([1.0, 0.5])

# Weights from input layer to hidden layer
W1 = torch.tensor([
    [0.2, 0.4, 0.6],
    [0.1, 0.3, 0.5]
])

# Hidden layer bias
b1 = torch.tensor([0.1, 0.1, 0.1])

# Calculate hidden layer
hidden = torch.relu(inputs @ W1 + b1)

# Weights from hidden layer to output layer
W2 = torch.tensor([
    [0.3],
    [0.5],
    [0.7]
])

# Output layer bias
b2 = torch.tensor([0.1])

# Calculate output
final_output = torch.sigmoid(hidden @ W2 + b2)

print("\nHidden layer output:", hidden)
print("Final network output:", final_output)

# Step 8: Calculate prediction loss

# Actual correct answer
target = torch.tensor([1.0])

# Calculate Binary Cross Entropy loss
loss = torch.nn.functional.binary_cross_entropy(
    final_output,
    target
)

print("\nActual target:", target.item())
print("Predicted output:", final_output.item())
print("Loss:", loss.item())

# Step 9: Backpropagation using PyTorch

# Create a trainable weight and bias
weight = torch.tensor(1.0, requires_grad=True)
bias = torch.tensor(0.0, requires_grad=True)

# Input and target
x = torch.tensor(2.0)
target_value = torch.tensor(10.0)

# Forward propagation
prediction = weight * x + bias

# Calculate Mean Squared Error
loss = (prediction - target_value) ** 2

# Backpropagation
loss.backward()

# Display gradients
print("\n--- Backpropagation ---")
print("Prediction:", prediction.item())
print("Loss:", loss.item())
print("Weight gradient:", weight.grad.item())
print("Bias gradient:", bias.grad.item())

# Step 10: Gradient Descent

learning_rate = 0.01

# Update weight and bias
with torch.no_grad():
    weight -= learning_rate * weight.grad
    bias -= learning_rate * bias.grad

# Make a new prediction
new_prediction = weight * x + bias

# Calculate new loss
new_loss = (new_prediction - target_value) ** 2

print("\n--- After Gradient Descent ---")
print("Updated weight:", weight.item())
print("Updated bias:", bias.item())
print("New prediction:", new_prediction.item())
print("New loss:", new_loss.item())

# Step 11: Training loop

# Reset weight and bias
weight = torch.tensor(1.0, requires_grad=True)
bias = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.01
epochs = 100

for epoch in range(epochs):

    # Forward propagation
    prediction = weight * x + bias

    # Calculate loss
    loss = (prediction - target_value) ** 2

    # Clear previous gradients
    if weight.grad is not None:
        weight.grad.zero_()

    if bias.grad is not None:
        bias.grad.zero_()

    # Backpropagation
    loss.backward()

    # Update parameters
    with torch.no_grad():
        weight -= learning_rate * weight.grad
        bias -= learning_rate * bias.grad

    # Display progress
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"Loss = {loss.item():.4f}"
        )

print("\nFinal prediction:", (weight * x + bias).item())

# Step 12: Plot training loss

import matplotlib.pyplot as plt

# Reset parameters
weight = torch.tensor(1.0, requires_grad=True)
bias = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.01
epochs = 100
loss_history = []

for epoch in range(epochs):

    prediction = weight * x + bias
    loss = (prediction - target_value) ** 2

    # Clear gradients
    if weight.grad is not None:
        weight.grad.zero_()

    if bias.grad is not None:
        bias.grad.zero_()

    # Backpropagation
    loss.backward()

    # Update parameters
    with torch.no_grad():
        weight -= learning_rate * weight.grad
        bias -= learning_rate * bias.grad

    # Record loss
    loss_history.append(loss.item())

# Plot graph
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()

# Step 13: Learning rate experiments

learning_rates = [0.01, 0.5, 5.0]
epochs = 20

for lr in learning_rates:

    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    print(f"\nLearning rate: {lr}")

    for epoch in range(epochs):

        prediction = w * x + b
        loss = (prediction - target_value) ** 2

        if w.grad is not None:
            w.grad.zero_()

        if b.grad is not None:
            b.grad.zero_()

        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

    print("Final prediction:", (w * x + b).item())
    print("Final loss:", loss.item())

# Step 14: Activation Functions

# ReLU
def relu(x):
    return torch.maximum(x, torch.tensor(0.0))

# Sigmoid
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Softmax
def softmax(x):
    exp_values = torch.exp(x - torch.max(x))
    return exp_values / torch.sum(exp_values)


# Test ReLU
relu_input = torch.tensor([-2.0, 0.0, 3.0])
print("\nReLU:", relu(relu_input))

# Test Sigmoid
sigmoid_input = torch.tensor(0.5)
print("Sigmoid:", sigmoid(sigmoid_input))

# Test Softmax
softmax_input = torch.tensor([2.0, 1.0, 0.1])
print("Softmax:", softmax(softmax_input))

# Step 15: Loss Functions

# Mean Squared Error
def mse_loss(actual, predicted):
    return torch.mean((actual - predicted) ** 2)


# Cross Entropy
def cross_entropy_loss(actual, predicted):
    epsilon = 1e-7
    predicted = torch.clamp(predicted, epsilon, 1 - epsilon)

    return -torch.mean(
        actual * torch.log(predicted)
        + (1 - actual) * torch.log(1 - predicted)
    )


# Test MSE
actual = torch.tensor([1.0, 0.0, 1.0])
predicted = torch.tensor([0.8, 0.2, 0.9])

print("\nMSE Loss:", mse_loss(actual, predicted).item())

# Test Binary Cross Entropy
print(
    "Cross Entropy Loss:",
    cross_entropy_loss(actual, predicted).item()
)

# Step 16: Dropout and Batch Normalization

import torch.nn as nn

# Create a simple neural network layer
class RegularizedNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(4, 8)
        self.batch_norm = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return torch.sigmoid(x)


# Create model
model = RegularizedNetwork()

# Use training mode
model.train()

# Test with a batch of 2 samples
sample_input = torch.randn(2, 4)

result = model(sample_input)

print("\nRegularized network output:", result)
print("Dropout:", model.dropout)
print("Batch Normalization:", model.batch_norm)

# Step 17: XOR Neural Network using PyTorch

import torch
import torch.nn as nn

# XOR input data
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

# Expected outputs
y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

# Create a 2-4-1 neural network
class XORNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


# Initialize model
torch.manual_seed(42)
model = XORNetwork()

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

# Train the model
epochs = 5000

for epoch in range(epochs):

    # Forward propagation
    predictions = model(X)

    # Calculate loss
    loss = criterion(predictions, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # Update weights
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")


# Test the trained network
model.eval()

with torch.no_grad():
    predictions = model(X)

print("\n--- XOR Predictions ---")

for i in range(len(X)):
    print(
        f"Input: {X[i].tolist()}, "
        f"Prediction: {predictions[i].item():.4f}, "
        f"Expected: {y[i].item():.0f}"
    )

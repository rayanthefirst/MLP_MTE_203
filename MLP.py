import numpy as np
import matplotlib.pyplot as plt

# Initialize weights and biases with random values
# np.random.seed(100)  # for reproducibility
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1
hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
output_bias = np.random.uniform(size=(1, outputLayerNeurons))


# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# XOR input and output
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])


# Learning rate and epochs
lr = 0.1
epochs = 10000

# Training algorithm
accuracy_data = []

for _ in range(epochs):
    # Forward pass
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    if _ % 100 == 0:
        accuracy_data.append(1 - np.mean(np.abs(outputs - predicted_output)))

    # Backward pass
    error = outputs - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr


# Testing the network on XOR inputs
# print(predicted_output)

epochs = [_ for _ in range(len(accuracy_data))]
plt.figure(figsize=(12, 6))
plt.plot(
    epochs,
    accuracy_data,
    marker="o",
    color="b",
    linestyle="-",
    linewidth=2,
    markersize=4,
)
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch Iteration at every 100th Iteration")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()


print(predicted_output, (1 - np.mean(np.abs(outputs - predicted_output))))

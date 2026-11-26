"""
MNIST Neural Network Training with NumPy (CPU Implementation)
Implements a two-layer fully connected network from scratch using NumPy
Demonstrates forward pass, backward pass, and weight updates without deep learning frameworks
"""

import numpy as np
import time

# Load MNIST dataset from binary files
# Use first 10000 training samples for faster training
X_train = np.fromfile("data/X_train.bin", dtype=np.float32).reshape(60000, 784)[:10000]
y_train = np.fromfile("data/y_train.bin", dtype=np.int32)[:10000]
X_test = np.fromfile("data/X_test.bin", dtype=np.float32).reshape(10000, 784)
y_test = np.fromfile("data/y_test.bin", dtype=np.int32)

# Normalize data using MNIST dataset statistics (mean and std computed from training set)
mean, std = 0.1307, 0.3081
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Reshape to (batch, channels, height, width) format
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

def relu(x):
    """
    ReLU activation function: f(x) = max(0, x)
    
    @param x: Input array
    @return: Output array with ReLU applied element-wise
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of ReLU activation function
    Returns 1 where x > 0, 0 otherwise
    
    @param x: Input array
    @return: Derivative array (1 where x > 0, 0 otherwise)
    """
    return (x > 0).astype(float)

def initialize_weights(input_size, output_size):
    """
    Initialize weights using He initialization (suitable for ReLU activations)
    Formula: weights ~ U(-sqrt(6/fan_in), sqrt(6/fan_in))
    
    @param input_size: Number of input features
    @param output_size: Number of output features
    @return: Weight matrix (input_size × output_size)
    """
    scale = np.sqrt(6.0 / input_size) 
    return (np.random.rand(input_size, output_size) * 2.0 - 1.0) * scale

def initialize_bias(output_size):
    """
    Initialize bias to zero
    
    @param output_size: Number of output features
    @return: Bias vector (1 × output_size)
    """
    return np.zeros((1, output_size))

def linear_forward(x, weights, bias):
    """
    Forward pass through a linear (fully connected) layer
    Computes: output = x @ weights + bias
    
    @param x: Input activations (batch_size × input_size)
    @param weights: Weight matrix (input_size × output_size)
    @param bias: Bias vector (1 × output_size)
    @return: Output activations (batch_size × output_size)
    """
    return x @ weights + bias

def linear_backward(grad_output, x, weights):
    """
    Backward pass through a linear layer
    Computes gradients with respect to weights, bias, and input
    
    Gradient formulas:
    - grad_weights = x^T @ grad_output
    - grad_bias = sum(grad_output, axis=0)
    - grad_input = grad_output @ weights^T
    
    @param grad_output: Gradient with respect to output (batch_size × output_size)
    @param x: Input activations from forward pass (batch_size × input_size)
    @param weights: Weight matrix (input_size × output_size)
    @return: Tuple of (grad_input, grad_weights, grad_bias)
    """
    grad_weights = x.T @ grad_output
    grad_bias = np.sum(grad_output, axis=0, keepdims=True)
    grad_input = grad_output @ weights.T
    return grad_input, grad_weights, grad_bias

def softmax(x):
    """
    Softmax activation function
    Formula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    
    @param x: Input logits (batch_size × num_classes)
    @return: Output probabilities (batch_size × num_classes)
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    """
    Compute cross-entropy loss between predictions and true labels
    
    Formula: loss = -mean(log(probabilities[true_labels]))
    
    @param y_pred: Predicted logits (batch_size × num_classes)
    @param y_true: True labels (batch_size,)
    @return: Average cross-entropy loss
    """
    batch_size = y_pred.shape[0]
    probabilities = softmax(y_pred)
    correct_log_probs = np.log(probabilities[np.arange(batch_size), y_true])
    loss = -np.sum(correct_log_probs) / batch_size
    return loss

class NeuralNetwork:
    """
    Two-layer fully connected neural network for MNIST digit classification
    Architecture: Input (784) -> Hidden (256, ReLU) -> Output (10)
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize neural network with random weights
        
        @param input_size: Number of input features (784 for MNIST)
        @param hidden_size: Number of hidden units (256)
        @param output_size: Number of output classes (10 for digits 0-9)
        """
        self.weights1 = initialize_weights(input_size, hidden_size)
        self.bias1 = initialize_bias(hidden_size)
        self.weights2 = initialize_weights(hidden_size, output_size)
        self.bias2 = initialize_bias(output_size)

    def forward(self, x):
        """
        Forward pass through the network
        
        @param x: Input batch (batch_size × 1 × 28 × 28)
        @return: Tuple of (output_logits, cache) where cache contains intermediate activations
        """
        batch_size = x.shape[0]
        # Flatten input: (batch_size, 1, 28, 28) -> (batch_size, 784)
        fc1_input = x.reshape(batch_size, -1)
        # First layer: linear transformation + ReLU
        fc1_output = linear_forward(fc1_input, self.weights1, self.bias1)
        relu_output = relu(fc1_output)
        # Second layer: linear transformation (no activation)
        fc2_output = linear_forward(relu_output, self.weights2, self.bias2)
        # Return output and cache for backward pass
        return fc2_output, (fc1_input, fc1_output, relu_output)

    def backward(self, grad_output, cache):
        """
        Backward pass through the network (backpropagation)
        Computes gradients with respect to all weights and biases
        
        @param grad_output: Gradient with respect to output (batch_size × output_size)
        @param cache: Tuple of intermediate activations from forward pass
        @return: Tuple of (grad_weights1, grad_bias1, grad_weights2, grad_bias2)
        """
        x, fc1_output, relu_output = cache

        # Backward through second layer
        grad_fc2, grad_weights2, grad_bias2 = linear_backward(grad_output, relu_output, self.weights2)
        # Backward through ReLU activation
        grad_relu = grad_fc2 * relu_derivative(fc1_output)
        # Backward through first layer
        grad_fc1, grad_weights1, grad_bias1 = linear_backward(grad_relu, x, self.weights1)
        return grad_weights1, grad_bias1, grad_weights2, grad_bias2

    def update_weights(self, grad_weights1, grad_bias1, grad_weights2, grad_bias2, learning_rate):
        """
        Update weights and biases using gradient descent
        Formula: weights = weights - learning_rate * grad_weights
        
        @param grad_weights1: Gradient for first layer weights
        @param grad_bias1: Gradient for first layer bias
        @param grad_weights2: Gradient for second layer weights
        @param grad_bias2: Gradient for second layer bias
        @param learning_rate: Learning rate for gradient descent
        """
        self.weights1 -= learning_rate * grad_weights1
        self.bias1 -= learning_rate * grad_bias1
        self.weights2 -= learning_rate * grad_weights2
        self.bias2 -= learning_rate * grad_bias2

def train_timed(model, X_train, y_train, X_test, y_test, batch_size, epochs, learning_rate):
    """
    Train the neural network with detailed timing instrumentation
    
    Tracks time spent on:
    - Data loading
    - Forward pass
    - Loss computation and gradient computation
    - Backward pass
    - Weight updates
    
    @param model: NeuralNetwork instance to train
    @param X_train: Training input data
    @param y_train: Training labels
    @param X_test: Test input data (unused, kept for compatibility)
    @param y_test: Test labels (unused, kept for compatibility)
    @param batch_size: Batch size for training
    @param epochs: Number of training epochs
    @param learning_rate: Learning rate for gradient descent
    """
    timing_stats = {
        'data_loading': 0.0,
        'forward': 0.0,
        'loss_computation': 0.0,
        'backward': 0.0,
        'weight_updates': 0.0,
        'total_time': 0.0
    }
    
    total_start = time.time()
    
    # Training loop over epochs
    for epoch in range(epochs):
        epoch_loss = 0.0
        # Process each batch
        for i in range(0, len(X_train), batch_size):
            # Time data loading
            data_start = time.time()
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            data_end = time.time()
            timing_stats['data_loading'] += data_end - data_start
            
            # Forward pass
            forward_start = time.time()
            y_pred, cache = model.forward(batch_X)
            forward_end = time.time()
            timing_stats['forward'] += forward_end - forward_start
            
            # Compute loss and output gradients
            loss_start = time.time()
            loss = cross_entropy_loss(y_pred, batch_y)
            epoch_loss += loss

            # Compute output gradients for backpropagation
            # Gradient of cross-entropy with softmax: grad = (softmax - one_hot) / batch_size
            softmax_probs = softmax(y_pred)
            y_true_one_hot = np.zeros_like(y_pred)
            y_true_one_hot[np.arange(len(batch_y)), batch_y] = 1
            grad_output = (softmax_probs - y_true_one_hot) / len(batch_y)
            loss_end = time.time()
            timing_stats['loss_computation'] += loss_end - loss_start

            # Backward pass: compute gradients
            backward_start = time.time()
            grad_weights1, grad_bias1, grad_weights2, grad_bias2 = model.backward(grad_output, cache)
            backward_end = time.time()
            timing_stats['backward'] += backward_end - backward_start
            
            # Update weights using computed gradients
            update_start = time.time()
            model.update_weights(grad_weights1, grad_bias1, grad_weights2, grad_bias2, learning_rate)
            update_end = time.time()
            timing_stats['weight_updates'] += update_end - update_start

        # Print average loss for this epoch
        print(f"Epoch {epoch} loss: {epoch_loss / (len(X_train) // batch_size):.4f}")

    # Calculate total training time
    total_end = time.time()
    timing_stats['total_time'] = total_end - total_start
    
    # Print detailed timing breakdown
    print("\n=== PYTHON NUMPY IMPLEMENTATION TIMING BREAKDOWN ===")
    print(f"Total training time: {timing_stats['total_time']:.1f} seconds\n")
    
    print("Detailed Breakdown:")
    print(f"  Data loading:     {timing_stats['data_loading']:6.3f}s ({100.0 * timing_stats['data_loading'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Forward pass:     {timing_stats['forward']:6.3f}s ({100.0 * timing_stats['forward'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Loss computation: {timing_stats['loss_computation']:6.3f}s ({100.0 * timing_stats['loss_computation'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Backward pass:    {timing_stats['backward']:6.3f}s ({100.0 * timing_stats['backward'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Weight updates:   {timing_stats['weight_updates']:6.3f}s ({100.0 * timing_stats['weight_updates'] / timing_stats['total_time']:5.1f}%)")
    
    print("Training completed!")

if __name__ == "__main__":
    # Network architecture parameters
    input_size = 784   # MNIST images: 28×28 = 784 pixels
    hidden_size = 256  # Number of hidden units
    output_size = 10   # Number of classes (digits 0-9)
    
    # Initialize neural network
    model = NeuralNetwork(input_size, hidden_size, output_size)
    
    # Training hyperparameters
    batch_size = 8
    epochs = 10
    learning_rate = 0.01
    
    # Train the model
    train_timed(model, X_train, y_train, X_test, y_test, batch_size, epochs, learning_rate)
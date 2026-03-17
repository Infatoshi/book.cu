"""
MNIST Neural Network Training with PyTorch CUDA
Implements a two-layer fully connected network for MNIST digit classification
Uses PyTorch's high-level API with CUDA acceleration and timing instrumentation
"""

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Training hyperparameters
TRAIN_SIZE = 10000      # Number of training samples to use
epochs = 10             # Number of training epochs
learning_rate = 1e-2    # Learning rate for SGD optimizer
batch_size = 8          # Batch size for training

# Set high precision for matrix multiplication (TensorFloat-32)
torch.set_float32_matmul_precision("high")

# Load MNIST dataset from binary files
# Data is stored as flattened 28x28 images (784 pixels per image)
X_train_np = np.fromfile("data/X_train.bin", dtype=np.float32).reshape(60000, 784)
y_train_np = np.fromfile("data/y_train.bin", dtype=np.int32)
X_test_np = np.fromfile("data/X_test.bin", dtype=np.float32).reshape(10000, 784)
y_test_np = np.fromfile("data/y_test.bin", dtype=np.int32)

# Normalize data using MNIST dataset statistics (mean and std computed from training set)
mean, std = 0.1307, 0.3081
X_train_np = (X_train_np - mean) / std
X_test_np = (X_test_np - mean) / std

# Convert numpy arrays to PyTorch tensors and move to GPU
# Reshape to (batch, channels, height, width) format for PyTorch
train_data = torch.from_numpy(X_train_np[:TRAIN_SIZE].reshape(-1, 1, 28, 28)).to("cuda")
train_labels = torch.from_numpy(y_train_np[:TRAIN_SIZE]).long().to("cuda")
test_data = torch.from_numpy(X_test_np.reshape(-1, 1, 28, 28)).to("cuda")
test_labels = torch.from_numpy(y_test_np).long().to("cuda")

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for MNIST digit classification
    Architecture: Input (784) -> Hidden (256) -> Output (10)
    Uses ReLU activation between layers
    """
    def __init__(self, in_features, hidden_features, num_classes):
        """
        Initialize the MLP network
        
        @param in_features: Number of input features (784 for MNIST)
        @param hidden_features: Number of hidden units (256)
        @param num_classes: Number of output classes (10 for digits 0-9)
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)  # First fully connected layer
        self.relu = nn.ReLU()                                # ReLU activation function
        self.fc2 = nn.Linear(hidden_features, num_classes)   # Second fully connected layer

    def forward(self, x):
        """
        Forward pass through the network
        
        @param x: Input tensor (batch_size × 1 × 28 × 28)
        @return: Output logits (batch_size × 10)
        """
        # Flatten input: (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.reshape(batch_size, 28 * 28)
        # First layer: linear transformation + ReLU activation
        x = self.fc1(x)
        x = self.relu(x)
        # Second layer: linear transformation (no activation, softmax applied in loss)
        x = self.fc2(x)
        return x


# Initialize model and move to GPU
model = MLP(in_features=784, hidden_features=256, num_classes=10).to("cuda")

# Initialize weights using He initialization (suitable for ReLU activations)
# Formula: weights ~ U(-sqrt(6/fan_in), sqrt(6/fan_in))
with torch.no_grad():
    # Initialize first layer weights
    fan_in = model.fc1.weight.size(1)  # Input dimension (784)
    scale = (6.0 / fan_in) ** 0.5
    model.fc1.weight.uniform_(-scale, scale)
    model.fc1.bias.zero_()  # Initialize biases to zero
    
    # Initialize second layer weights
    fan_in = model.fc2.weight.size(1)  # Hidden dimension (256)
    scale = (6.0 / fan_in) ** 0.5
    model.fc2.weight.uniform_(-scale, scale)
    model.fc2.bias.zero_()  # Initialize biases to zero

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Combines LogSoftmax and NLLLoss
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Stochastic Gradient Descent


def train_timed(model, criterion, optimizer, epoch, timing_stats, epoch_losses):
    """
    Train the model for one epoch with detailed timing instrumentation
    
    Tracks time spent on:
    - Data loading
    - Forward pass
    - Loss computation
    - Backward pass (gradient computation)
    - Weight updates
    
    @param model: PyTorch model to train
    @param criterion: Loss function
    @param optimizer: Optimizer for weight updates
    @param epoch: Current epoch number (unused, kept for compatibility)
    @param timing_stats: Dictionary to accumulate timing statistics
    @param epoch_losses: List to store average loss per epoch
    """
    model.train()  # Set model to training mode
    epoch_loss = 0.0
    
    # Calculate number of iterations per epoch
    iters_per_epoch = math.ceil(train_data.shape[0] / batch_size)
    
    # Process each batch
    for i in range(iters_per_epoch):
        # Time data loading
        data_start = time.time()
        data = train_data[i * batch_size : (i + 1) * batch_size]
        target = train_labels[i * batch_size : (i + 1) * batch_size]
        data_end = time.time()
        timing_stats['data_loading'] += data_end - data_start
        
        # Zero out gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        forward_start = time.time()
        outputs = model(data)
        forward_end = time.time()
        timing_stats['forward'] += forward_end - forward_start
        
        # Compute loss
        loss_start = time.time()
        loss = criterion(outputs, target)
        epoch_loss += loss.item()
        loss_end = time.time()
        timing_stats['loss_computation'] += loss_end - loss_start
        
        # Backward pass: compute gradients
        backward_start = time.time()
        loss.backward()
        backward_end = time.time()
        timing_stats['backward'] += backward_end - backward_start
        
        # Update weights using computed gradients
        update_start = time.time()
        optimizer.step()
        optimizer.zero_grad()  # Zero gradients for next iteration
        update_end = time.time()
        timing_stats['weight_updates'] += update_end - update_start
    
    # Store average loss for this epoch
    epoch_losses.append(epoch_loss / iters_per_epoch)


def evaluate(model, test_data, test_labels):
    """
    Evaluate model accuracy on test dataset
    
    Computes average batch accuracy across all test batches
    Uses no_grad context to disable gradient computation for efficiency
    
    @param model: PyTorch model to evaluate
    @param test_data: Test input data tensor
    @param test_labels: Test labels tensor
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    total_batch_accuracy = torch.tensor(0.0, device=device)
    num_batches = 0

    # Disable gradient computation during evaluation
    with torch.no_grad():
        for i in range(len(test_data)):
            # Get batch data
            data = test_data[i * batch_size : (i + 1) * batch_size]
            target = test_labels[i * batch_size : (i + 1) * batch_size]
            
            # Forward pass to get predictions
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class indices
            
            # Compute accuracy for this batch
            correct_batch = (predicted == target).sum().item()
            total_batch = target.size(0)
            if total_batch != 0:  
                batch_accuracy = correct_batch / total_batch
                total_batch_accuracy += batch_accuracy
                num_batches += 1

    # Compute and print average accuracy
    avg_batch_accuracy = total_batch_accuracy / num_batches
    print(f"Average Batch Accuracy: {avg_batch_accuracy * 100:.2f}%")


if __name__ == "__main__":
    # Initialize timing statistics dictionary
    timing_stats = {
        'data_loading': 0.0,
        'forward': 0.0,
        'loss_computation': 0.0,
        'backward': 0.0,
        'weight_updates': 0.0,
        'total_time': 0.0
    }
    epoch_losses = []
    
    # Start total training timer
    total_start = time.time()
    
    # Training loop over epochs
    for epoch in range(epochs):
        train_timed(model, criterion, optimizer, epoch, timing_stats, epoch_losses)
        print(f"Epoch {epoch} loss: {epoch_losses[epoch]:.4f}")

    # Calculate total training time
    total_end = time.time()
    timing_stats['total_time'] = total_end - total_start
    
    # Print detailed timing breakdown
    print("\n=== PYTORCH CUDA IMPLEMENTATION TIMING BREAKDOWN ===")
    print(f"Total training time: {timing_stats['total_time']:.1f} seconds\n")
    
    print("Detailed Breakdown:")
    print(f"  Data loading:     {timing_stats['data_loading']:6.3f}s ({100.0 * timing_stats['data_loading'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Forward pass:     {timing_stats['forward']:6.3f}s ({100.0 * timing_stats['forward'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Loss computation: {timing_stats['loss_computation']:6.3f}s ({100.0 * timing_stats['loss_computation'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Backward pass:    {timing_stats['backward']:6.3f}s ({100.0 * timing_stats['backward'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Weight updates:   {timing_stats['weight_updates']:6.3f}s ({100.0 * timing_stats['weight_updates'] / timing_stats['total_time']:5.1f}%)")

    print("Finished Training")

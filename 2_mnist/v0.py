"""
MNIST Data Preprocessing Script
Downloads MNIST dataset and saves it in binary format for efficient loading
"""

import os
import shutil

import numpy as np
from torchvision import datasets, transforms

# Directory to save processed data
save_dir = "data"
os.makedirs(save_dir, exist_ok=True)

# Transform to convert PIL images to tensors
transform = transforms.Compose([transforms.ToTensor()])

# Download MNIST training dataset
mnist_train = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

# Download MNIST test dataset
mnist_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# Extract and normalize image data (28x28 pixels flattened to 784)
# Normalize pixel values from [0, 255] to [0.0, 1.0]
X_train = mnist_train.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_train = mnist_train.targets.numpy().astype(np.int32)
X_test = mnist_test.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_test = mnist_test.targets.numpy().astype(np.int32)

# Save data and labels to binary files for fast loading
X_train.tofile(os.path.join(save_dir, "X_train.bin"))
y_train.tofile(os.path.join(save_dir, "y_train.bin"))
X_test.tofile(os.path.join(save_dir, "X_test.bin"))
y_test.tofile(os.path.join(save_dir, "y_test.bin"))

# Write metadata file with dataset information
with open(os.path.join(save_dir, "metadata.txt"), "w") as f:
    f.write(f"Training samples: {X_train.shape[0]}\n")
    f.write(f"Test samples: {X_test.shape[0]}\n")
    f.write(f"Input dimensions: {X_train.shape[1]}\n")
    f.write(f"Number of classes: {len(np.unique(y_train))}\n")

# Clean up temporary directories created by torchvision
if os.path.exists("mnist_data"):
    shutil.rmtree("mnist_data")
    print("Removed mnist_data directory")

if os.path.exists(os.path.join(save_dir, "MNIST")):
    shutil.rmtree(os.path.join(save_dir, "MNIST"))
    print("Removed MNIST raw data directory")

print("MNIST dataset has been downloaded and saved in binary format in data/ directory.")

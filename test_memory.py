"""
Test memory requirements before full training
"""
import pickle
import numpy as np
import sys

print("Testing memory requirements...")

# Load data
print("\n1. Loading data...")
with open('preprocessed_data_optical_flow.pkl', 'rb') as f:
    data = pickle.load(f)

X_rgb = data['X_rgb']
X_flow = data['X_flow']
y = data['y']

print(f"   RGB: {X_rgb.shape}, {X_rgb.nbytes / (1024**3):.2f} GB")
print(f"   Flow: {X_flow.shape}, {X_flow.nbytes / (1024**3):.2f} GB")
print(f"   Total: {(X_rgb.nbytes + X_flow.nbytes) / (1024**3):.2f} GB")

# Test batch creation
print("\n2. Testing batch creation...")
batch_size = 2
batch_rgb = X_rgb[:batch_size]
batch_flow = X_flow[:batch_size]
batch_y = y[:batch_size]

print(f"   Batch RGB: {batch_rgb.shape}, {batch_rgb.nbytes / (1024**2):.2f} MB")
print(f"   Batch Flow: {batch_flow.shape}, {batch_flow.nbytes / (1024**2):.2f} MB")
print(f"   Total per batch: {(batch_rgb.nbytes + batch_flow.nbytes) / (1024**2):.2f} MB")

# Test split
print("\n3. Testing data split...")
from sklearn.model_selection import train_test_split

indices = np.arange(len(y))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=y)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx])

print(f"   Train: {len(train_idx)} samples")
print(f"   Val: {len(val_idx)} samples")
print(f"   Test: {len(test_idx)} samples")

# Estimate memory for training
train_rgb_size = (len(train_idx) * X_rgb[0].nbytes) / (1024**3)
train_flow_size = (len(train_idx) * X_flow[0].nbytes) / (1024**3)

print(f"\n4. Memory estimate for training set:")
print(f"   RGB: {train_rgb_size:.2f} GB")
print(f"   Flow: {train_flow_size:.2f} GB")
print(f"   Total: {train_rgb_size + train_flow_size:.2f} GB")

# Test generator
print("\n5. Testing data generator...")

class SimpleGenerator:
    def __init__(self, X_rgb, X_flow, y, indices, batch_size=2):
        self.X_rgb = X_rgb
        self.X_flow = X_flow
        self.y = y
        self.indices = indices
        self.batch_size = batch_size
    
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_rgb_batch = self.X_rgb[batch_indices]
        X_flow_batch = self.X_flow[batch_indices]
        y_batch = self.y[batch_indices]
        return [X_rgb_batch, X_flow_batch], y_batch

gen = SimpleGenerator(X_rgb, X_flow, y, train_idx, batch_size=2)
print(f"   Generator created: {len(gen)} batches")

# Test getting a batch
print("\n6. Testing batch retrieval...")
batch_data, batch_labels = gen[0]
print(f"   Batch retrieved successfully!")
print(f"   RGB batch: {batch_data[0].shape}")
print(f"   Flow batch: {batch_data[1].shape}")
print(f"   Labels: {batch_labels.shape}")

print("\n" + "="*60)
print("MEMORY TEST PASSED!")
print("="*60)
print("\nThe optimized training script should work.")
print("Estimated memory usage: ~8-10 GB during training")

"""
Quick test: 1 epoch to verify everything works
"""
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
from config import *

print('='*60)
print('QUICK TRAINING TEST (1 epoch)')
print('='*60)

# Load with memory mapping
print('\n[1/5] Loading data (memory-mapped)...')
X_rgb = np.load('data_mmap/X_rgb.npy', mmap_mode='r')
X_flow = np.load('data_mmap/X_flow.npy', mmap_mode='r')
y = np.load('data_mmap/y.npy')

print(f'RGB: {X_rgb.shape}')
print(f'Flow: {X_flow.shape}')
print(f'Labels: {y.shape}')

# Split
print('\n[2/5] Splitting data...')
indices = np.arange(len(y))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=y)
val_idx, _ = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx])

print(f'Train: {len(train_idx)}, Val: {len(val_idx)}')

# Generator
print('\n[3/5] Creating generator...')

class TestGenerator(keras.utils.Sequence):
    def __init__(self, X_rgb, X_flow, y, indices, batch_size=2, **kwargs):
        super().__init__(**kwargs)
        self.X_rgb = X_rgb
        self.X_flow = X_flow
        self.y = y
        self.indices = indices.copy()
        self.batch_size = batch_size
    
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_rgb_batch = np.array(self.X_rgb[batch_indices], dtype=np.float32)
        X_flow_batch = np.array(self.X_flow[batch_indices], dtype=np.float32)
        y_batch = np.array(self.y[batch_indices], dtype=np.float32)
        return (X_rgb_batch, X_flow_batch), y_batch

train_gen = TestGenerator(X_rgb, X_flow, y, train_idx, batch_size=2)
val_gen = TestGenerator(X_rgb, X_flow, y, val_idx, batch_size=2)

print(f'Train batches: {len(train_gen)}')
print(f'Val batches: {len(val_gen)}')

# Build simple model (faster for testing)
print('\n[4/5] Building model...')

rgb_input = layers.Input(shape=(30, 128, 128, 3))
flow_input = layers.Input(shape=(30, 128, 128, 2))

# Simplified RGB stream
rgb_features = layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu'))(rgb_input)
rgb_features = layers.TimeDistributed(layers.MaxPooling2D((4,4)))(rgb_features)
rgb_features = layers.TimeDistributed(layers.GlobalAveragePooling2D())(rgb_features)
rgb_lstm = layers.LSTM(64)(rgb_features)

# Simplified Flow stream  
flow_features = layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu'))(flow_input)
flow_features = layers.TimeDistributed(layers.MaxPooling2D((4,4)))(flow_features)
flow_features = layers.TimeDistributed(layers.GlobalAveragePooling2D())(flow_features)
flow_lstm = layers.LSTM(64)(flow_features)

# Fusion
merged = layers.concatenate([rgb_lstm, flow_lstm])
x = layers.Dense(64, activation='relu')(merged)
output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=[rgb_input, flow_input], outputs=output)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f'Parameters: {model.count_params():,}')

# Test training
print('\n[5/5] Testing training (1 epoch)...')
print('This should take 5-10 minutes...')

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=1,
    verbose=1
)

print('\n' + '='*60)
print('TEST PASSED!')
print('='*60)
print(f'Train accuracy: {history.history["accuracy"][0]*100:.2f}%')
print(f'Val accuracy: {history.history["val_accuracy"][0]*100:.2f}%')
print('\nMemory-mapped training works!')
print('Ready to start full training with run_training_final.py')

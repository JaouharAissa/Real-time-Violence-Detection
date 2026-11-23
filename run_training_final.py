"""
Violence Detection - Final Memory-Mapped Training
Works with limited RAM by using memory-mapped numpy arrays
"""

import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from config import *

print('=' * 60)
print('VIOLENCE DETECTION - MEMORY-MAPPED TRAINING')
print('=' * 60)

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# System Check
# ============================================================

print('\n[System Check]')
print(f'TensorFlow: {tf.__version__}')

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f'GPU: {len(gpus)} detected')
else:
    print('GPU: None (CPU training)')

try:
    tf.config.optimizer.set_jit(True)
    print('XLA: Enabled')
except:
    print('XLA: Not available')

# ============================================================
# Load Data with Memory Mapping
# ============================================================

print('\n[1/7] Loading data (memory-mapped - no RAM usage)...')

# Load with mmap_mode='r' - reads from disk, doesn't load into RAM
X_rgb = np.load('data_mmap/X_rgb.npy', mmap_mode='r')
X_flow = np.load('data_mmap/X_flow.npy', mmap_mode='r')
y = np.load('data_mmap/y.npy')

print(f'RGB: {X_rgb.shape}')
print(f'Flow: {X_flow.shape}')
print(f'Labels: {y.shape}')
print(f'Violence: {np.sum(y == 1)}, Non-Violence: {np.sum(y == 0)}')
print('Memory-mapped: Data stays on disk, loaded only when needed')

# ============================================================
# Split Data
# ============================================================

print('\n[2/7] Splitting dataset...')

indices = np.arange(len(y))
train_idx, temp_idx = train_test_split(
    indices, test_size=0.3, random_state=42, stratify=y
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx]
)

print(f'Train: {len(train_idx)} ({len(train_idx)/len(y)*100:.1f}%)')
print(f'Val: {len(val_idx)} ({len(val_idx)/len(y)*100:.1f}%)')
print(f'Test: {len(test_idx)} ({len(test_idx)/len(y)*100:.1f}%)')

# Class weights
y_train = y[train_idx]
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f'Class weights: {class_weight_dict}')

# ============================================================
# Data Generator
# ============================================================

print('\n[3/7] Creating data generators...')

class MemoryMappedGenerator(keras.utils.Sequence):
    """Generator that reads from memory-mapped arrays"""
    
    def __init__(self, X_rgb, X_flow, y, indices, batch_size=2, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.X_rgb = X_rgb
        self.X_flow = X_flow
        self.y = y
        self.indices = indices.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Load only this batch from disk
        X_rgb_batch = np.array(self.X_rgb[batch_indices], dtype=np.float32)
        X_flow_batch = np.array(self.X_flow[batch_indices], dtype=np.float32)
        y_batch = np.array(self.y[batch_indices], dtype=np.float32)
        
        return (X_rgb_batch, X_flow_batch), y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

train_gen = MemoryMappedGenerator(X_rgb, X_flow, y, train_idx, batch_size=2, shuffle=True)
val_gen = MemoryMappedGenerator(X_rgb, X_flow, y, val_idx, batch_size=2, shuffle=False)
test_gen = MemoryMappedGenerator(X_rgb, X_flow, y, test_idx, batch_size=2, shuffle=False)

print(f'Train: {len(train_gen)} batches')
print(f'Val: {len(val_gen)} batches')
print(f'Test: {len(test_gen)} batches')

# ============================================================
# Build Model
# ============================================================

print('\n[4/7] Building Two-Stream model...')

def build_two_stream_model():
    # RGB Stream
    rgb_input = layers.Input(shape=(30, 128, 128, 3), name='rgb_input')
    
    resnet_base = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(128, 128, 3),
        pooling=None
    )
    resnet_base.trainable = False
    
    rgb_features = layers.TimeDistributed(resnet_base)(rgb_input)
    rgb_features = layers.TimeDistributed(layers.GlobalAveragePooling2D())(rgb_features)
    rgb_lstm = layers.LSTM(256, return_sequences=False)(rgb_features)
    rgb_lstm = layers.Dropout(0.5)(rgb_lstm)
    
    # Flow Stream
    flow_input = layers.Input(shape=(30, 128, 128, 2), name='flow_input')
    
    flow_features = layers.TimeDistributed(
        layers.Conv2D(64, (3, 3), activation='relu', padding='same')
    )(flow_input)
    flow_features = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(flow_features)
    
    flow_features = layers.TimeDistributed(
        layers.Conv2D(128, (3, 3), activation='relu', padding='same')
    )(flow_features)
    flow_features = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(flow_features)
    
    flow_features = layers.TimeDistributed(
        layers.Conv2D(256, (3, 3), activation='relu', padding='same')
    )(flow_features)
    flow_features = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(flow_features)
    flow_features = layers.TimeDistributed(layers.GlobalAveragePooling2D())(flow_features)
    
    flow_lstm = layers.LSTM(128, return_sequences=False)(flow_features)
    flow_lstm = layers.Dropout(0.5)(flow_lstm)
    
    # Fusion
    merged = layers.concatenate([rgb_lstm, flow_lstm])
    x = layers.Dense(256, activation='relu')(merged)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = models.Model(inputs=[rgb_input, flow_input], outputs=output)
    return model, resnet_base

model, resnet_base = build_two_stream_model()
print(f'Parameters: {model.count_params():,}')

# ============================================================
# Phase 1: Frozen ResNet50
# ============================================================

print('\n[5/7] Phase 1: Training (frozen ResNet50, 10 epochs)...')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

callbacks_phase1 = [
    ModelCheckpoint(
        os.path.join(MODELS_DIR, f'phase1_{timestamp}.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

history_phase1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weight_dict,
    callbacks=callbacks_phase1,
    verbose=1
)

print('Phase 1 complete!')

# ============================================================
# Phase 2: Fine-tuning
# ============================================================

print('\n[6/7] Phase 2: Fine-tuning (10 epochs)...')

resnet_base.trainable = True
for layer in resnet_base.layers[:-30]:
    layer.trainable = False

print(f'Unfrozen: {sum([1 for l in resnet_base.layers if l.trainable])}/{len(resnet_base.layers)} layers')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

callbacks_phase2 = [
    ModelCheckpoint(
        os.path.join(MODELS_DIR, f'phase2_{timestamp}.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

history_phase2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weight_dict,
    callbacks=callbacks_phase2,
    verbose=1
)

print('Phase 2 complete!')

# ============================================================
# Evaluation
# ============================================================

print('\n[7/7] Evaluating...')

test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen, verbose=0)

print(f'\n=== Test Results ===')
print(f'Accuracy: {test_acc*100:.2f}%')
print(f'Precision: {test_prec*100:.2f}%')
print(f'Recall: {test_rec*100:.2f}%')
print(f'F1-Score: {2*(test_prec*test_rec)/(test_prec+test_rec)*100:.2f}%')

# Save final model
final_path = os.path.join(MODELS_DIR, f'violence_detector_{timestamp}.keras')
model.save(final_path)

print(f'\nModel saved: {final_path}')

print('\n' + '=' * 60)
print('TRAINING COMPLETE!')
print('=' * 60)
print(f'Test Accuracy: {test_acc*100:.2f}%')
print('=' * 60)

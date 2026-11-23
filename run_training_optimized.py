"""
Violence Detection - Memory-Optimized Training
Uses data generators to minimize memory usage
"""

import tensorflow as tf
import numpy as np
import pickle
import os
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from config import *

print('=' * 60)
print('VIOLENCE DETECTION - MEMORY-OPTIMIZED TRAINING')
print('=' * 60)

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# GPU Configuration
# ============================================================

print('\n[GPU Configuration]')
print(f'TensorFlow version: {tf.__version__}')

gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {len(gpus)}')

if len(gpus) > 0:
    print('GPU detected!')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print('No GPU - training on CPU')

# XLA Optimization
try:
    tf.config.optimizer.set_jit(True)
    print('XLA enabled')
except:
    pass

# ============================================================
# Load Data Indices Only
# ============================================================

print('\n[1/7] Loading data indices...')

# Load file to get indices
with open('preprocessed_data_optical_flow.pkl', 'rb') as f:
    data = pickle.load(f)

y = data['y']
n_samples = len(y)

print(f'Total samples: {n_samples}')
print(f'Violence: {np.sum(y == 1)}, Non-Violence: {np.sum(y == 0)}')

# Split indices
indices = np.arange(n_samples)
train_idx, temp_idx = train_test_split(
    indices, test_size=0.3, random_state=42, stratify=y
)

val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx]
)

print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')

# Class weights
y_train = y[train_idx]
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f'Class weights: {class_weight_dict}')

# ============================================================
# Data Generator
# ============================================================

print('\n[2/7] Creating data generators...')

class TwoStreamGenerator(keras.utils.Sequence):
    """Memory-efficient data generator"""
    
    def __init__(self, X_rgb, X_flow, y, indices, batch_size=2, shuffle=True):
        self.X_rgb = X_rgb
        self.X_flow = X_flow
        self.y = y
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        X_rgb_batch = self.X_rgb[batch_indices]
        X_flow_batch = self.X_flow[batch_indices]
        y_batch = self.y[batch_indices]
        
        return [X_rgb_batch, X_flow_batch], y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Create generators
train_gen = TwoStreamGenerator(
    data['X_rgb'], data['X_flow'], data['y'],
    train_idx, batch_size=2, shuffle=True
)

val_gen = TwoStreamGenerator(
    data['X_rgb'], data['X_flow'], data['y'],
    val_idx, batch_size=2, shuffle=False
)

test_gen = TwoStreamGenerator(
    data['X_rgb'], data['X_flow'], data['y'],
    test_idx, batch_size=2, shuffle=False
)

print(f'Train batches: {len(train_gen)}')
print(f'Val batches: {len(val_gen)}')
print(f'Test batches: {len(test_gen)}')

# ============================================================
# Build Model
# ============================================================

print('\n[3/7] Building Two-Stream model...')

def build_two_stream_model(rgb_shape, flow_shape):
    # RGB Stream
    rgb_input = layers.Input(shape=rgb_shape, name='rgb_input')
    
    resnet_base = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(rgb_shape[1], rgb_shape[2], rgb_shape[3]),
        pooling=None
    )
    resnet_base.trainable = False
    
    rgb_features = layers.TimeDistributed(resnet_base)(rgb_input)
    rgb_features = layers.TimeDistributed(layers.GlobalAveragePooling2D())(rgb_features)
    rgb_lstm = layers.LSTM(256, return_sequences=False)(rgb_features)
    rgb_lstm = layers.Dropout(0.5)(rgb_lstm)
    
    # Flow Stream
    flow_input = layers.Input(shape=flow_shape, name='flow_input')
    
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

rgb_shape = (30, 128, 128, 3)
flow_shape = (30, 128, 128, 2)

model, resnet_base = build_two_stream_model(rgb_shape, flow_shape)
print(f'Model built! Parameters: {model.count_params():,}')

# ============================================================
# Phase 1: Frozen ResNet50
# ============================================================

print('\n[4/7] Phase 1: Training with frozen ResNet50...')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
checkpoint_path = os.path.join(MODELS_DIR, f'two_stream_phase1_{timestamp}.keras')

callbacks_phase1 = [
    ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    TensorBoard(log_dir=os.path.join(LOGS_DIR, f'phase1_{timestamp}'))
]

print('Training Phase 1 (10 epochs)...')
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

print('\n[5/7] Phase 2: Fine-tuning ResNet50...')

resnet_base.trainable = True
for layer in resnet_base.layers[:-30]:
    layer.trainable = False

print(f'Unfrozen layers: {sum([1 for layer in resnet_base.layers if layer.trainable])}/{len(resnet_base.layers)}')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

checkpoint_path_phase2 = os.path.join(MODELS_DIR, f'two_stream_final_{timestamp}.keras')

callbacks_phase2 = [
    ModelCheckpoint(checkpoint_path_phase2, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=1),
    TensorBoard(log_dir=os.path.join(LOGS_DIR, f'phase2_{timestamp}'))
]

print('Training Phase 2 (10 epochs)...')
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

print('\n[6/7] Evaluating on test set...')

test_loss, test_acc, test_precision, test_recall = model.evaluate(test_gen, verbose=0)

print(f'\n=== Test Results ===')
print(f'Loss: {test_loss:.4f}')
print(f'Accuracy: {test_acc*100:.2f}%')
print(f'Precision: {test_precision*100:.2f}%')
print(f'Recall: {test_recall*100:.2f}%')
print(f'F1-Score: {2*(test_precision*test_recall)/(test_precision+test_recall)*100:.2f}%')

# ============================================================
# Save Model
# ============================================================

print('\n[7/7] Saving final model...')

final_model_path = os.path.join(MODELS_DIR, f'violence_detector_two_stream_{timestamp}.keras')
model.save(final_model_path)

print(f'Model saved: {final_model_path}')

# Save history
history_path = os.path.join(MODELS_DIR, f'training_history_{timestamp}.pkl')
with open(history_path, 'wb') as f:
    pickle.dump({
        'phase1': history_phase1.history,
        'phase2': history_phase2.history,
        'test_metrics': {
            'loss': test_loss,
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall
        }
    }, f)

print(f'History saved: {history_path}')

print('\n' + '=' * 60)
print('TRAINING COMPLETE!')
print('=' * 60)
print(f'\nFinal Model: {final_model_path}')
print(f'Test Accuracy: {test_acc*100:.2f}%')
print('=' * 60)

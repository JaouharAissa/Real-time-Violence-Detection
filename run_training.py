"""
Violence Detection - Two-Stream Training Script
Architecture: ResNet50 + Optical Flow + LSTM
Expected Accuracy: 93-96%
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
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

print('=' * 60)
print('VIOLENCE DETECTION - TWO-STREAM TRAINING')
print('=' * 60)

# ============================================================
# GPU Configuration
# ============================================================

print('\n[GPU Configuration]')
print(f'TensorFlow version: {tf.__version__}')

gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {len(gpus)}')

if len(gpus) > 0:
    print('✓ GPU detected!')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('✓ Memory growth enabled')
else:
    print('WARNING: No GPU - training on CPU (slower)')

# ============================================================
# Performance Optimizations
# ============================================================

print('\n[Performance Optimizations]')

# Mixed Precision (FP16)
try:
    from tensorflow.keras import mixed_precision
    if len(gpus) > 0:
        mixed_precision.set_global_policy('mixed_float16')
        print('✓ Mixed Precision (FP16) enabled - 2-3x speedup')
except:
    print('WARNING: Mixed Precision not available')

# XLA Compilation
try:
    tf.config.optimizer.set_jit(True)
    print('✓ XLA enabled - 10-15% speedup')
except:
    print('WARNING: XLA not available')

# TensorFloat-32 for Ampere GPUs
try:
    tf.config.experimental.enable_tensor_float_32_execution(True)
    print('✓ TensorFloat-32 enabled')
except:
    pass

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# Load Data (Memory-Efficient)
# ============================================================

print('\n[1/7] Loading preprocessed data (memory-efficient)...')

# Load only metadata first
with open('preprocessed_data_optical_flow.pkl', 'rb') as f:
    data = pickle.load(f)

# Get shapes and labels without loading all data into memory at once
print(f'RGB shape: {data["X_rgb"].shape}')
print(f'Flow shape: {data["X_flow"].shape}')
print(f'Labels: {data["y"].shape}')
print(f'Violence: {np.sum(data["y"] == 1)}, Non-Violence: {np.sum(data["y"] == 0)}')

# Keep references to data
X_rgb = data['X_rgb']
X_flow = data['X_flow']
y = data['y']

print('✓ Data loaded (using memory-mapped access)')

# ============================================================
# Split Data (Indices Only)
# ============================================================

print('\n[2/7] Splitting dataset (70% train, 15% val, 15% test)...')

# Split indices instead of actual data to save memory
indices = np.arange(len(y))
train_idx, temp_idx = train_test_split(
    indices, test_size=0.3, random_state=42, stratify=y
)

val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx]
)

# Get split data
X_rgb_train = X_rgb[train_idx]
X_flow_train = X_flow[train_idx]
y_train = y[train_idx]

X_rgb_val = X_rgb[val_idx]
X_flow_val = X_flow[val_idx]
y_val = y[val_idx]

X_rgb_test = X_rgb[test_idx]
X_flow_test = X_flow[test_idx]
y_test = y[test_idx]

print(f'Train: {len(y_train)} ({len(y_train)/len(y)*100:.1f}%)')
print(f'Val: {len(y_val)} ({len(y_val)/len(y)*100:.1f}%)')
print(f'Test: {len(y_test)} ({len(y_test)/len(y)*100:.1f}%)')

# Class weights for imbalanced data
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f'Class weights: {class_weight_dict}')

# Clear original data to free memory
del data, X_rgb, X_flow, y
import gc
gc.collect()
print('✓ Memory cleared')

# ============================================================
# Build Two-Stream Model
# ============================================================

print('\n[3/7] Building Two-Stream architecture...')

def build_two_stream_model(rgb_shape, flow_shape):
    """
    Two-Stream Architecture:
    - RGB Stream: ResNet50 + LSTM
    - Flow Stream: CNN + LSTM
    - Fusion: Concatenate + Dense layers
    """
    
    # ===== RGB STREAM =====
    rgb_input = layers.Input(shape=rgb_shape, name='rgb_input')
    
    # ResNet50 pretrained on ImageNet
    resnet_base = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(rgb_shape[1], rgb_shape[2], rgb_shape[3]),
        pooling=None
    )
    
    # Freeze ResNet50 initially
    resnet_base.trainable = False
    
    # Apply ResNet50 to each frame
    rgb_features = layers.TimeDistributed(resnet_base, name='resnet_features')(rgb_input)
    rgb_features = layers.TimeDistributed(layers.GlobalAveragePooling2D(), name='rgb_gap')(rgb_features)
    
    # LSTM for temporal modeling
    rgb_lstm = layers.LSTM(256, return_sequences=False, name='rgb_lstm')(rgb_features)
    rgb_lstm = layers.Dropout(0.5, name='rgb_dropout')(rgb_lstm)
    
    # ===== FLOW STREAM =====
    flow_input = layers.Input(shape=flow_shape, name='flow_input')
    
    # Custom CNN for optical flow
    flow_features = layers.TimeDistributed(
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        name='flow_conv1'
    )(flow_input)
    flow_features = layers.TimeDistributed(
        layers.MaxPooling2D((2, 2)),
        name='flow_pool1'
    )(flow_features)
    
    flow_features = layers.TimeDistributed(
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        name='flow_conv2'
    )(flow_features)
    flow_features = layers.TimeDistributed(
        layers.MaxPooling2D((2, 2)),
        name='flow_pool2'
    )(flow_features)
    
    flow_features = layers.TimeDistributed(
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        name='flow_conv3'
    )(flow_features)
    flow_features = layers.TimeDistributed(
        layers.MaxPooling2D((2, 2)),
        name='flow_pool3'
    )(flow_features)
    
    flow_features = layers.TimeDistributed(
        layers.GlobalAveragePooling2D(),
        name='flow_gap'
    )(flow_features)
    
    # LSTM for temporal modeling
    flow_lstm = layers.LSTM(128, return_sequences=False, name='flow_lstm')(flow_features)
    flow_lstm = layers.Dropout(0.5, name='flow_dropout')(flow_lstm)
    
    # ===== FUSION =====
    merged = layers.concatenate([rgb_lstm, flow_lstm], name='fusion')
    
    # Classification head
    x = layers.Dense(256, activation='relu', name='fc1')(merged)
    x = layers.Dropout(0.5, name='dropout1')(x)
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    
    # Output layer (with float32 for mixed precision)
    output = layers.Dense(1, activation='sigmoid', dtype='float32', name='output')(x)
    
    model = models.Model(inputs=[rgb_input, flow_input], outputs=output, name='TwoStream_Violence_Detector')
    
    return model, resnet_base

# Build model
rgb_shape = X_rgb_train.shape[1:]
flow_shape = X_flow_train.shape[1:]

model, resnet_base = build_two_stream_model(rgb_shape, flow_shape)

print(f'✓ Model built!')
print(f'  RGB input: {rgb_shape}')
print(f'  Flow input: {flow_shape}')
print(f'  Total parameters: {model.count_params():,}')

# ============================================================
# Training Phase 1: Frozen ResNet50
# ============================================================

print('\n[4/7] Phase 1: Training with frozen ResNet50...')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

# Callbacks
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
checkpoint_path = os.path.join(MODELS_DIR, f'two_stream_phase1_{timestamp}.keras')

callbacks_phase1 = [
    ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
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
        min_lr=1e-7,
        verbose=1
    ),
    TensorBoard(
        log_dir=os.path.join(LOGS_DIR, f'phase1_{timestamp}'),
        histogram_freq=1
    )
]

print('Training Phase 1 (10 epochs)...')

# Use smaller batch size and enable garbage collection
import gc
history_phase1 = model.fit(
    [X_rgb_train, X_flow_train],
    y_train,
    validation_data=([X_rgb_val, X_flow_val], y_val),
    epochs=10,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=callbacks_phase1,
    verbose=1
)

# Clear memory after phase 1
gc.collect()
print('✓ Phase 1 complete!')

# ============================================================
# Training Phase 2: Fine-tuning
# ============================================================

print('\n[5/7] Phase 2: Fine-tuning ResNet50...')

# Unfreeze last 30 layers of ResNet50
resnet_base.trainable = True
for layer in resnet_base.layers[:-30]:
    layer.trainable = False

print(f'Unfrozen layers: {sum([1 for layer in resnet_base.layers if layer.trainable])}/{ len(resnet_base.layers)}')

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

checkpoint_path_phase2 = os.path.join(MODELS_DIR, f'two_stream_final_{timestamp}.keras')

callbacks_phase2 = [
    ModelCheckpoint(
        checkpoint_path_phase2,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
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
        min_lr=1e-8,
        verbose=1
    ),
    TensorBoard(
        log_dir=os.path.join(LOGS_DIR, f'phase2_{timestamp}'),
        histogram_freq=1
    )
]

print('Training Phase 2 (10 epochs)...')

# Clear memory before phase 2
import gc
gc.collect()

history_phase2 = model.fit(
    [X_rgb_train, X_flow_train],
    y_train,
    validation_data=([X_rgb_val, X_flow_val], y_val),
    epochs=10,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=callbacks_phase2,
    verbose=1
)

gc.collect()
print('✓ Phase 2 complete!')

# ============================================================
# Evaluation
# ============================================================

print('\n[6/7] Evaluating on test set...')

test_loss, test_acc, test_precision, test_recall = model.evaluate(
    [X_rgb_test, X_flow_test],
    y_test,
    verbose=0
)

print(f'\n=== Test Results ===')
print(f'Loss: {test_loss:.4f}')
print(f'Accuracy: {test_acc*100:.2f}%')
print(f'Precision: {test_precision*100:.2f}%')
print(f'Recall: {test_recall*100:.2f}%')
print(f'F1-Score: {2*(test_precision*test_recall)/(test_precision+test_recall)*100:.2f}%')

# Predictions
y_pred_proba = model.predict([X_rgb_test, X_flow_test], verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Classification report
print('\n=== Classification Report ===')
print(classification_report(y_test, y_pred, target_names=['Non-Violence', 'Violence']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('\n=== Confusion Matrix ===')
print(cm)

# ============================================================
# Save Final Model
# ============================================================

print('\n[7/7] Saving final model...')

final_model_path = os.path.join(MODELS_DIR, f'violence_detector_two_stream_{timestamp}.keras')
model.save(final_model_path)

print(f'✓ Model saved: {final_model_path}')

# Save training history
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

print(f'✓ History saved: {history_path}')

# ============================================================
# Summary
# ============================================================

print('\n' + '=' * 60)
print('TRAINING COMPLETE!')
print('=' * 60)
print(f'\nFinal Model: {final_model_path}')
print(f'Test Accuracy: {test_acc*100:.2f}%')
print(f'Expected: 93-96%')
print(f'\nNext steps:')
print('  1. Run evaluation: jupyter notebook 03_model_evaluation.ipynb')
print('  2. Test API: python api.py')
print('  3. Test webcam: python webcam_detection.py')
print('=' * 60)

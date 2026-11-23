"""
Evaluate the trained Two-Stream model
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

print('='*60)
print('MODEL EVALUATION')
print('='*60)

# Load data
print('\n[1/4] Loading test data...')
X_rgb = np.load('data_mmap/X_rgb.npy', mmap_mode='r')
X_flow = np.load('data_mmap/X_flow.npy', mmap_mode='r')
y = np.load('data_mmap/y.npy')

# Use same split as training
from sklearn.model_selection import train_test_split
indices = np.arange(len(y))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=y)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx])

X_rgb_test = X_rgb[test_idx]
X_flow_test = X_flow[test_idx]
y_test = y[test_idx]

print(f'Test samples: {len(y_test)}')
print(f'Violence: {np.sum(y_test == 1)}, Non-Violence: {np.sum(y_test == 0)}')

# Load model
print('\n[2/4] Loading trained model...')
model = keras.models.load_model('models/phase2_20260415_184926.keras')
print(f'Model loaded: {model.count_params():,} parameters')

# Predictions
print('\n[3/4] Making predictions...')
y_pred_proba = model.predict([X_rgb_test, X_flow_test], batch_size=2, verbose=1)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Metrics
print('\n[4/4] Calculating metrics...')
accuracy = accuracy_score(y_test, y_pred)
print(f'\n{"="*60}')
print(f'TEST ACCURACY: {accuracy*100:.2f}%')
print(f'{"="*60}')

print('\n=== Classification Report ===')
print(classification_report(y_test, y_pred, target_names=['Non-Violence', 'Violence']))

print('\n=== Confusion Matrix ===')
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f'\nTrue Negatives: {cm[0,0]}')
print(f'False Positives: {cm[0,1]}')
print(f'False Negatives: {cm[1,0]}')
print(f'True Positives: {cm[1,1]}')

# Save confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Violence', 'Violence'],
            yticklabels=['Non-Violence', 'Violence'])
plt.title(f'Confusion Matrix - Accuracy: {accuracy*100:.2f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print('\n✓ Confusion matrix saved: confusion_matrix.png')

print('\n' + '='*60)
print('EVALUATION COMPLETE!')
print('='*60)
print(f'\nFinal Accuracy: {accuracy*100:.2f}%')
print(f'Model: models/phase2_20260415_184926.keras')
print('='*60)

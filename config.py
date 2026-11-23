"""Configuration file for the violence detection project"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIOLENCE_DIR = os.path.join(BASE_DIR, 'Violence')
NON_VIOLENCE_DIR = os.path.join(BASE_DIR, 'NonViolence')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Video processing parameters
FRAME_HEIGHT = 128
FRAME_WIDTH = 128
SEQUENCE_LENGTH = 30  # Number of frames per video sequence
CHANNELS = 3

# Training parameters
BATCH_SIZE = 2  # Reduced for memory constraints
EPOCHS = 50
LEARNING_RATE = 0.0001
K_FOLDS = 5  # For cross-validation (if used)

# Model parameters
DROPOUT_RATE = 0.5
LSTM_UNITS = 256

# Classes
CLASSES = ['NonViolence', 'Violence']
NUM_CLASSES = len(CLASSES)

# Training strategy
USE_CROSS_VALIDATION = False  # Set to True for K-Fold, False for Train/Val/Test
USE_DATA_AUGMENTATION = True  # Highly recommended
USE_TRANSFER_LEARNING = False  # Set to True for better performance (slower)

# Data split ratios (if not using cross-validation)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

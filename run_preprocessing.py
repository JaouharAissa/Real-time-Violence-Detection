"""
Violence Detection - Data Preprocessing with Optical Flow
Automated execution script
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from config import *

print('=' * 60)
print('VIOLENCE DETECTION - PREPROCESSING')
print('=' * 60)
print(f'\nDataset:')
print(f'  Violence videos: {len(os.listdir(VIOLENCE_DIR))}')
print(f'  Non-Violence videos: {len(os.listdir(NON_VIOLENCE_DIR))}')
print()

# ============================================================
# 1. Video Integrity Check
# ============================================================

def check_video_integrity(video_path):
    """Vérifie si une vidéo est lisible et valide"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            cap.release()
            return False, 'Cannot open video'
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames == 0 or fps == 0 or width == 0 or height == 0:
            cap.release()
            return False, 'Invalid metadata'
        
        # Vérifier quelques frames
        frames_to_check = min(5, total_frames)
        readable_frames = 0
        
        for i in range(frames_to_check):
            ret, frame = cap.read()
            if ret and frame is not None:
                readable_frames += 1
        
        cap.release()
        
        if readable_frames < frames_to_check * 0.5:
            return False, 'Too many unreadable frames'
        
        return True, {'frames': total_frames, 'fps': fps, 'width': width, 'height': height}
        
    except Exception as e:
        return False, str(e)


def scan_dataset():
    """Scanner tous les vidéos"""
    results = []
    
    print('Scanning Violence videos...')
    for filename in tqdm(os.listdir(VIOLENCE_DIR)):
        if filename.endswith(('.mp4', '.avi')):
            video_path = os.path.join(VIOLENCE_DIR, filename)
            is_valid, info = check_video_integrity(video_path)
            results.append({
                'filename': filename,
                'path': video_path,
                'class': 'Violence',
                'valid': is_valid,
                'info': info
            })
    
    print('Scanning Non-Violence videos...')
    for filename in tqdm(os.listdir(NON_VIOLENCE_DIR)):
        if filename.endswith(('.mp4', '.avi')):
            video_path = os.path.join(NON_VIOLENCE_DIR, filename)
            is_valid, info = check_video_integrity(video_path)
            results.append({
                'filename': filename,
                'path': video_path,
                'class': 'NonViolence',
                'valid': is_valid,
                'info': info
            })
    
    return pd.DataFrame(results)


print('\n[1/6] Scanning dataset...')
df_videos = scan_dataset()

print(f'\nTotal videos: {len(df_videos)}')
print(f'Valid videos: {df_videos["valid"].sum()}')
print(f'Corrupted videos: {(~df_videos["valid"]).sum()}')

# ============================================================
# 2. Extract Metadata
# ============================================================

print('\n[2/6] Extracting metadata...')
valid_videos = df_videos[df_videos['valid']].copy()

valid_videos['frames'] = valid_videos['info'].apply(lambda x: x['frames'] if isinstance(x, dict) else 0)
valid_videos['fps'] = valid_videos['info'].apply(lambda x: x['fps'] if isinstance(x, dict) else 0)
valid_videos['duration'] = valid_videos['frames'] / valid_videos['fps']

print('\nStatistics:')
print(valid_videos[['frames', 'fps', 'duration']].describe())

# ============================================================
# 3. Remove Outliers
# ============================================================

print('\n[3/6] Removing outliers...')

def remove_outliers(df, column, n_std=3):
    mean = df[column].mean()
    std = df[column].std()
    lower_bound = mean - n_std * std
    upper_bound = mean + n_std * std
    mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    removed = len(df) - mask.sum()
    print(f'  {column}: Removed {removed} outliers')
    return df[mask]

clean_videos = valid_videos.copy()
initial_count = len(clean_videos)
clean_videos = remove_outliers(clean_videos, 'frames', n_std=3)
clean_videos = remove_outliers(clean_videos, 'duration', n_std=3)

print(f'\nRemaining: {len(clean_videos)}/{initial_count}')
print(f'Violence: {(clean_videos["class"] == "Violence").sum()}')
print(f'NonViolence: {(clean_videos["class"] == "NonViolence").sum()}')

# ============================================================
# 4. Optical Flow Functions
# ============================================================

def compute_optical_flow(prev_gray, curr_gray):
    """Calcule l'Optical Flow dense avec Farneback"""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    
    # Convertir flow en magnitude et angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Normaliser magnitude [0, 1]
    magnitude = np.clip(magnitude / 20.0, 0, 1)
    
    # Créer représentation 2-channel (magnitude + angle normalisé)
    angle_norm = angle / (2 * np.pi)
    
    flow_representation = np.stack([magnitude, angle_norm], axis=-1)
    
    return flow_representation.astype(np.float32)


def extract_rgb_and_flow(video_path, max_frames=SEQUENCE_LENGTH):
    """Extrait RGB frames + Optical Flow"""
    rgb_frames = []
    flow_frames = []
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None, None
    
    # Sample frames uniformly
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    
    prev_gray = None
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret or frame is None or frame.size == 0:
            continue
        
        # Resize RGB
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.astype(np.float32) / 255.0
        
        # Compute optical flow
        curr_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            flow = compute_optical_flow(prev_gray, curr_gray)
            flow_frames.append(flow)
        
        rgb_frames.append(frame_rgb)
        prev_gray = curr_gray
    
    cap.release()
    
    # Vérifier validité
    if len(rgb_frames) < max_frames * 0.5:
        return None, None
    
    # Pour le premier frame, dupliquer le premier flow
    if len(flow_frames) > 0:
        flow_frames.insert(0, flow_frames[0].copy())
    
    # Padding
    while len(rgb_frames) < max_frames:
        rgb_frames.append(rgb_frames[-1].copy())
        if len(flow_frames) > 0:
            flow_frames.append(flow_frames[-1].copy())
    
    while len(flow_frames) < max_frames:
        if len(flow_frames) > 0:
            flow_frames.append(flow_frames[-1].copy())
        else:
            flow_frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 2), dtype=np.float32))
    
    return np.array(rgb_frames[:max_frames]), np.array(flow_frames[:max_frames])

# ============================================================
# 5. Load Dataset with Optical Flow
# ============================================================

print('\n[4/6] Loading dataset with Optical Flow...')
print('⚠️  This will take 2-3 hours (calculating optical flow for 2000 videos)...')

def load_dataset_with_flow(video_list):
    """Charge RGB + Optical Flow pour tous les vidéos"""
    X_rgb = []
    X_flow = []
    y = []
    failed = []
    
    for idx, row in tqdm(video_list.iterrows(), total=len(video_list), desc='Processing videos'):
        video_path = row['path']
        label = 1 if row['class'] == 'Violence' else 0
        
        rgb, flow = extract_rgb_and_flow(video_path)
        
        if rgb is not None and flow is not None:
            X_rgb.append(rgb)
            X_flow.append(flow)
            y.append(label)
        else:
            failed.append(row['filename'])
    
    if len(failed) > 0:
        print(f'\n⚠️  Failed: {len(failed)} videos')
    
    return np.array(X_rgb), np.array(X_flow), np.array(y), failed


X_rgb, X_flow, y, failed = load_dataset_with_flow(clean_videos)

print(f'\n=== Dataset Final ===')
print(f'RGB shape: {X_rgb.shape}')
print(f'Flow shape: {X_flow.shape}')
print(f'Labels shape: {y.shape}')
print(f'Violence: {np.sum(y == 1)}')
print(f'Non-Violence: {np.sum(y == 0)}')

# ============================================================
# 6. Quality Check
# ============================================================

print('\n[5/6] Quality check...')

print('\n=== RGB Data Quality ===')
print(f'Min: {X_rgb.min():.4f}, Max: {X_rgb.max():.4f}')
print(f'Mean: {X_rgb.mean():.4f}, Std: {X_rgb.std():.4f}')
print(f'NaN: {np.isnan(X_rgb).sum()}, Inf: {np.isinf(X_rgb).sum()}')

print('\n=== Optical Flow Quality ===')
print(f'Min: {X_flow.min():.4f}, Max: {X_flow.max():.4f}')
print(f'Mean: {X_flow.mean():.4f}, Std: {X_flow.std():.4f}')
print(f'NaN: {np.isnan(X_flow).sum()}, Inf: {np.isinf(X_flow).sum()}')

# Analyser l'intensité du mouvement par classe
violence_flow = X_flow[y == 1, :, :, :, 0].mean()
nonviolence_flow = X_flow[y == 0, :, :, :, 0].mean()

print(f'\n=== Movement Intensity ===')
print(f'Violence avg flow: {violence_flow:.4f}')
print(f'Non-Violence avg flow: {nonviolence_flow:.4f}')
print(f'Ratio: {violence_flow/nonviolence_flow:.2f}x')
print('✓ Les vidéos de violence devraient avoir plus de mouvement!')

# ============================================================
# 7. Save Data
# ============================================================

print('\n[6/6] Saving preprocessed data...')

data_to_save = {
    'X_rgb': X_rgb,
    'X_flow': X_flow,
    'y': y,
    'video_info': clean_videos[['filename', 'class', 'frames', 'fps', 'duration']].to_dict('records'),
    'failed_videos': failed,
    'preprocessing_params': {
        'sequence_length': SEQUENCE_LENGTH,
        'frame_height': FRAME_HEIGHT,
        'frame_width': FRAME_WIDTH,
        'rgb_channels': 3,
        'flow_channels': 2,
        'optical_flow_method': 'Farneback'
    }
}

with open('preprocessed_data_optical_flow.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

file_size = os.path.getsize('preprocessed_data_optical_flow.pkl') / (1024**3)
print(f'✓ Data saved! Size: {file_size:.2f} GB')

print('\n' + '=' * 60)
print('🎯 PREPROCESSING COMPLETE!')
print('=' * 60)
print(f'\nOutput file: preprocessed_data_optical_flow.pkl')
print(f'Total videos processed: {len(X_rgb)}')
print(f'Violence: {np.sum(y == 1)}, Non-Violence: {np.sum(y == 0)}')
print(f'\nNext step: Run training')
print('  python run_training.py')
print('=' * 60)

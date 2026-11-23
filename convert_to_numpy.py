"""
Convert pickle to separate numpy files for memory-mapped access
This allows training without loading 18GB into RAM
"""
import pickle
import numpy as np
import os

print("="*60)
print("CONVERTING DATA TO MEMORY-MAPPED FORMAT")
print("="*60)

print("\n[1/4] Loading pickle file...")
print("This will take 5-10 minutes...")

with open('preprocessed_data_optical_flow.pkl', 'rb') as f:
    data = pickle.load(f)

X_rgb = data['X_rgb']
X_flow = data['X_flow']
y = data['y']

print(f"Loaded: RGB {X_rgb.shape}, Flow {X_flow.shape}, Labels {y.shape}")

# Create directory
os.makedirs('data_mmap', exist_ok=True)

print("\n[2/4] Saving RGB data...")
np.save('data_mmap/X_rgb.npy', X_rgb)
print(f"Saved: data_mmap/X_rgb.npy ({os.path.getsize('data_mmap/X_rgb.npy')/(1024**3):.2f} GB)")

print("\n[3/4] Saving Flow data...")
np.save('data_mmap/X_flow.npy', X_flow)
print(f"Saved: data_mmap/X_flow.npy ({os.path.getsize('data_mmap/X_flow.npy')/(1024**3):.2f} GB)")

print("\n[4/4] Saving labels...")
np.save('data_mmap/y.npy', y)
print(f"Saved: data_mmap/y.npy ({os.path.getsize('data_mmap/y.npy')/(1024**2):.2f} MB)")

print("\n" + "="*60)
print("CONVERSION COMPLETE!")
print("="*60)
print("\nYou can now delete preprocessed_data_optical_flow.pkl to save space")
print("The new files use memory-mapped access (no RAM loading)")

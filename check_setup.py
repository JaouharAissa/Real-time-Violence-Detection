"""Check if the environment is properly set up"""

import sys

print("=" * 60)
print("🔍 Checking Violence Detection Project Setup")
print("=" * 60)

# Check Python version
print(f"\n✓ Python version: {sys.version}")

# Check required packages
packages = {
    'tensorflow': '2.15.0',
    'keras': '2.15.0',
    'cv2': 'opencv-python',
    'numpy': '1.24.3',
    'pandas': '2.1.4',
    'sklearn': 'scikit-learn',
    'matplotlib': '3.8.2',
    'seaborn': '0.13.0',
    'fastapi': '0.109.0',
    'uvicorn': '0.27.0'
}

print("\n📦 Checking packages:")
missing = []
for package, version in packages.items():
    try:
        if package == 'cv2':
            import cv2
            print(f"  ✓ opencv-python: {cv2.__version__}")
        elif package == 'sklearn':
            import sklearn
            print(f"  ✓ scikit-learn: {sklearn.__version__}")
        else:
            module = __import__(package)
            ver = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {package}: {ver}")
    except ImportError:
        print(f"  ✗ {package}: NOT INSTALLED")
        missing.append(version if '.' in version else package)

# Check GPU
print("\n🖥️  Checking GPU:")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  ✓ GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"    - {gpu.name}")
    else:
        print("  ⚠️  No GPU detected (training will be slower)")
except Exception as e:
    print(f"  ✗ Error checking GPU: {e}")

# Check directories
print("\n📁 Checking directories:")
import os
dirs = ['Violence', 'NonViolence', 'models', 'logs', 'temp']
for d in dirs:
    exists = os.path.exists(d)
    status = "✓" if exists else "✗"
    count = ""
    if exists and d in ['Violence', 'NonViolence']:
        files = [f for f in os.listdir(d) if f.endswith(('.mp4', '.avi'))]
        count = f" ({len(files)} videos)"
    print(f"  {status} {d}/{count}")

# Summary
print("\n" + "=" * 60)
if missing:
    print("❌ Setup incomplete!")
    print(f"\nInstall missing packages:")
    print(f"pip install {' '.join(missing)}")
else:
    print("✅ Setup complete! Ready to start training.")
    print("\nNext steps:")
    print("1. Run: jupyter notebook")
    print("2. Open: 01_data_preprocessing.ipynb")
    print("3. Execute all cells")
print("=" * 60)

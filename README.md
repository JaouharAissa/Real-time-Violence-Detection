# 🎯 Violence Detection - Two-Stream Deep Learning

Système de détection de violence en temps réel utilisant une architecture Two-Stream (ResNet50 + Optical Flow + LSTM).

## 🏆 Architecture

### Two-Stream Network
- **RGB Stream**: ResNet50 (ImageNet pré-entraîné) + LSTM
- **Flow Stream**: CNN + LSTM pour Optical Flow
- **Fusion**: Concatenation + Dense layers
- **Performance attendue**: 93-96% accuracy

### Pourquoi Optical Flow?
L'Optical Flow capture les mouvements entre frames, essentiel pour détecter:
- Mouvements rapides (coups, chutes)
- Gestes agressifs
- Changements brusques de mouvement

## 📊 Dataset

- **Total**: 2000 vidéos
- **Violence**: 1000 vidéos
- **Non-Violence**: 1000 vidéos
- **Split**: 70% train, 15% validation, 15% test

## 🚀 Quick Start

### Option 1: Automatic Installation (Recommended)

**Windows:**
```bash
install.bat
```

**Linux/Mac:**
```bash
chmod +x install.sh
./install.sh
```

### Option 2: Manual Installation

```bash
pip install tensorflow opencv-python tqdm jupyter ipykernel scikit-learn matplotlib seaborn fastapi uvicorn python-multipart pillow
```

### Verify Installation

```bash
python check_setup.py
```

## ⚡ GPU Optimization

Le projet inclut des optimisations GPU avancées pour accélérer l'entraînement de **3-4x**:

- **Mixed Precision (FP16)**: 2-3x plus rapide sur RTX/Tesla GPUs
- **XLA Compilation**: 10-15% speedup supplémentaire
- **TensorFloat-32**: Accélération automatique sur Ampere GPUs
- **Memory Growth**: Prévient les erreurs Out-of-Memory

**Temps d'entraînement estimé:**
- Avec GPU optimisé: ~2-3 heures
- Sans GPU: ~10-15 heures

Voir `GPU_OPTIMIZATION_GUIDE.md` pour plus de détails.

## 📝 Utilisation

### 1. Preprocessing (Nettoyage + Optical Flow)

```bash
jupyter notebook 01_data_preprocessing_optical_flow.ipynb
```

Ce notebook:
- Vérifie l'intégrité des vidéos
- Supprime les vidéos corrompues et outliers
- Extrait les frames RGB
- Calcule l'Optical Flow (Farneback)
- Sauvegarde: `preprocessed_data_optical_flow.pkl`

### 2. Entraînement Two-Stream (avec GPU optimisé)

```bash
jupyter notebook 02_model_training_optical_flow.ipynb
```

Entraînement en 2 phases:
- **Phase 1** (20 epochs): ResNet50 gelé
- **Phase 2** (30 epochs): Fine-tuning des 30 dernières couches

Sauvegarde: `models/final_model_two_stream.h5`

### 3. Évaluation

```bash
jupyter notebook 03_model_evaluation.ipynb
```

Analyse détaillée:
- Confusion matrix
- ROC curve
- Classification report
- Métriques par classe

### 4. Détection Webcam en Temps Réel

```bash
python webcam_detection.py
```

Contrôles:
- `q` - Quitter
- `s` - Sauvegarder le frame actuel

## 📁 Structure du Projet

```
.
├── Violence/                    # 1000 vidéos de violence
├── NonViolence/                 # 1000 vidéos non-violence
├── models/                      # Modèles sauvegardés
│   ├── phase1_two_stream.h5
│   ├── phase2_two_stream.h5
│   └── final_model_two_stream.h5
├── logs/                        # Logs et visualisations
├── temp/                        # Fichiers temporaires
├── 01_data_preprocessing_optical_flow.ipynb
├── 02_model_training_optical_flow.ipynb
├── 03_model_evaluation.ipynb
├── config.py                    # Configuration
├── utils.py                     # Fonctions utilitaires
├── app.py                       # API FastAPI
├── webcam_detection.py          # Détection webcam
├── requirements.txt             # Dépendances
└── TRAINING_STRATEGY.md         # Stratégie détaillée
```

## 🎨 Optical Flow Visualization

L'Optical Flow est visualisé en heatmap:
- **Rouge/Jaune**: Mouvement intense (violence probable)
- **Bleu/Vert**: Mouvement faible (non-violence)

## ⚙️ Configuration

Modifiez `config.py` pour ajuster:
- `SEQUENCE_LENGTH`: Nombre de frames par vidéo (défaut: 30)
- `FRAME_HEIGHT`, `FRAME_WIDTH`: Résolution (défaut: 128x128)
- `BATCH_SIZE`: Taille du batch (défaut: 8)
- `EPOCHS`: Nombre d'epochs (défaut: 50)

## 📈 Performance Attendue

| Métrique | Valeur |
|----------|--------|
| Accuracy | 93-96% |
| Precision | 92-95% |
| Recall | 91-94% |
| AUC | 0.95-0.98 |

## 🔬 Comparaison des Approches

| Approche | Accuracy | Temps |
|----------|----------|-------|
| ResNet50 + LSTM | 90-92% | 3-4h |
| **Two-Stream (RGB + Flow)** | **93-96%** | **4-5h** |
| 3D CNN (I3D) | 94-97% | 15-20h |

## 🛠️ Techniques Utilisées

### Data Augmentation
- Horizontal flip (50%)
- Brightness/Contrast (70%)
- Temporal dropout (20%)
- Cohérent pour RGB et Flow

### Transfer Learning
- ResNet50 pré-entraîné sur ImageNet
- Fine-tuning progressif

### Régularisation
- Dropout (0.3-0.5)
- Batch Normalization
- Class weights pour déséquilibre

### Callbacks
- EarlyStopping (patience=10-15)
- ReduceLROnPlateau (patience=5-7)
- ModelCheckpoint (save best)
- TensorBoard


## 🐛 Troubleshooting

### Erreur: Model not found
```bash
# Entraîner le modèle d'abord
jupyter notebook 02_model_training_optical_flow.ipynb
```

### Erreur: Out of memory
```python
# Réduire BATCH_SIZE dans config.py
BATCH_SIZE = 4  # au lieu de 8
```

### Webcam ne s'ouvre pas
```python
# Vérifier l'index de la caméra
cap = cv2.VideoCapture(1)  # Essayer 1 au lieu de 0
```

## 📚 Références

- Two-Stream CNNs: Simonyan & Zisserman (2014)
- Optical Flow: Farneback (2003)
- ResNet50: He et al. (2015)
- Violence Detection: état de l'art


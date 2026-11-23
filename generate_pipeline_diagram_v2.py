import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuration de la figure
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis('off')
ax.set_xlim(0, 18)
ax.set_ylim(0, 8)

# Fonction améliorée pour dessiner des boîtes parfaites
def draw_box(ax, center_x, center_y, width, height, text, facecolor, edgecolor):
    # Les coordonnées X, Y du coin inférieur gauche
    x = center_x - width / 2
    y = center_y - height / 2
    
    box = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        ec=edgecolor, fc=facecolor, lw=2.5, alpha=0.9
    )
    ax.add_patch(box)
    
    # Texte bien centré
    ax.text(center_x, center_y, text, 
            ha='center', va='center', 
            fontsize=11, fontweight='bold', color='#2c3e50', 
            linespacing=1.5)

# Fonction améliorée pour les flèches (ne rentrent pas dans les boîtes)
def draw_arrow(ax, start_x, start_y, end_x, end_y):
    ax.annotate('', 
                xy=(end_x, end_y), 
                xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle="-|>,head_width=0.6,head_length=0.8", 
                                color="#34495e", lw=2.5, shrinkA=0, shrinkB=0))

# --- DÉFINITION DES BLOCS ---
W, H = 3.2, 1.4  # Largeur et Hauteur des boîtes

# 1. Vidéo Brute
x1, y1 = 2.0, 4.0
draw_box(ax, x1, y1, W, H, "Vidéo Brute\n(Séquence MP4 / WebCam)", '#ecf0f1', '#95a5a6')

# 2. Extraction
x2, y2 = 6.5, 4.0
draw_box(ax, x2, y2, W, H, "Extraction des Frames\n(30 images par séquence)", '#bdc3c7', '#7f8c8d')

# 3a. RGB
x3a, y3a = 11.5, 6.0
draw_box(ax, x3a, y3a, W, H, "Flux Spatial (RGB)\nRedimensionnement\n(224 x 224 x 3)", '#3498db', '#2980b9')

# 3b. Optical Flow
x3b, y3b = 11.5, 2.0
draw_box(ax, x3b, y3b, W, H, "Flux Temporel (Optical Flow)\nAlgorithme Farneback\n(224 x 224 x 2)", '#e74c3c', '#c0392b')

# 4. Modèle
x4, y4 = 16.0, 4.0
draw_box(ax, x4, y4, W+0.2, H, "Modèle Two-Stream\n(Fusion LSTM)", '#2ecc71', '#27ae60')

# --- DÉFINITION DES FLÈCHES ---
# Vidéo -> Extraction
draw_arrow(ax, x1 + W/2 + 0.1, y1, x2 - W/2 - 0.1, y2)

# Extraction -> RGB
# Départ milieu-droit de l'extraction, arrivée milieu-gauche du RGB
draw_arrow(ax, x2 + W/2 + 0.1, y2, x3a - W/2 - 0.1, y3a)

# Extraction -> Flow
draw_arrow(ax, x2 + W/2 + 0.1, y2, x3b - W/2 - 0.1, y3b)

# RGB -> Modèle
draw_arrow(ax, x3a + W/2 + 0.1, y3a, x4 - W/2 - 0.2, y4 + 0.2)

# Flow -> Modèle
draw_arrow(ax, x3b + W/2 + 0.1, y3b, x4 - W/2 - 0.2, y4 - 0.2)

# Titre
plt.title("Pipeline de Traitement des Données Vidéo", fontsize=18, fontweight='bold', y=0.9)

# Sauvegarde
import os
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/pipeline_diagram.png', dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
plt.close()

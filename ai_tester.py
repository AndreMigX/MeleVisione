import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# 1. SETUP DEL DISPOSITIVO (IL TUO M4 PRO!)
# PyTorch su Mac Apple Silicon usa 'mps' (Metal Performance Shaders) invece di 'cuda'
if torch.backends.mps.is_available():
    device = "mps"
    print("🔥 Accelerazione Hardware M4 (MPS) ATTIVATA! 🔥")
else:
    device = "cpu"
    print("⚠️ Attenzione: MPS non trovato, sto usando la CPU.")

# 2. CARICAMENTO DEL MODELLO SAM
print("Caricamento di SAM in corso...")
sam_checkpoint = "./sam_vit_b_01ec64.pth"
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Creiamo il generatore automatico di maschere
mask_generator = SamAutomaticMaskGenerator(sam)

# 3. LETTURA DELL'IMMAGINE
IMAGE_PATH = "./images/frame_0000.jpg" # <--- CAMBIA QUESTO!
print(f"Leggo l'immagine {IMAGE_PATH}...")
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 4. ESTRAZIONE DELLE MASCHERE (Qui l'M4 lavora)
print("SAM sta processando l'immagine (trovando tutti gli oggetti)...")
masks = mask_generator.generate(image_rgb)
print(f"Fatto! SAM ha trovato {len(masks)} oggetti/segmenti in questa foto.")

# 5. VISUALIZZAZIONE DEL RISULTATO
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

plt.figure(figsize=(10,10))
plt.imshow(image_rgb)
show_anns(masks)
plt.axis('off')
plt.title(f"Segment Anything (M4 Pro) - {len(masks)} maschere")
plt.show()
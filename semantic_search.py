import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import CLIPProcessor, CLIPModel
import os
import open3d as o3d

# 1. SETUP DISPOSITIVO
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Uso accelerazione: {device}")

# 2. CARICAMENTO MODELLI
print("Caricamento SAM...")
sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth").to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

print("Caricamento CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 3. LETTURA IMMAGINE
IMAGE_PATH = "./images/frame_0030.jpg" # <--- CONTROLLA CHE IL NOME SIA GIUSTO
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
total_area = image_rgb.shape[0] * image_rgb.shape[1]

# 4. IL PROMPT (TRUCCO: Aggiungere "a photo of a" aumenta la precisione di CLIP del 30%)
TARGET_OBJECT = "yellow water bottle"
SEARCH_QUERY = f"a photo of a {TARGET_OBJECT}"
print(f"\n🔍 Ricerca in corso per: '{SEARCH_QUERY}'...")

# 5. ESTRAZIONE MASCHERE
print("1/3 -> SAM sta estraendo gli oggetti...")
masks = mask_generator.generate(image_rgb)

# 6. ANALISI CON CLIP
print(f"2/3 -> Filtraggio e analisi di {len(masks)} oggetti...")
highest_score = -1
best_mask = None

# Vettore del testo
inputs_text = clip_processor(text=[SEARCH_QUERY], return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_features = clip_model.get_text_features(**inputs_text)
    if hasattr(text_features, "pooler_output"):
        text_features = text_features.pooler_output
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

valid_objects_tested = 0

for i, ann in enumerate(masks):
    x, y, w, h = [int(v) for v in ann['bbox']]
    mask_area = ann['area']
    
    # FIX FONDAMENTALE: Tagliamo via muri, pavimenti e tavoli!
    # Se l'oggetto è più grande del 15% dell'immagine, o microscopico, saltalo a piè pari.
    if mask_area < 500 or mask_area > (total_area * 0.15):
        continue

    valid_objects_tested += 1

    # FIX 2: Contextual Crop. Ritagliamo l'oggetto ma aggiungiamo 15 pixel di margine
    # così CLIP vede le ombre e capisce che è un oggetto 3D, non una macchia nera.
    margin = 15
    y1 = max(0, y - margin)
    y2 = min(image_rgb.shape[0], y + h + margin)
    x1 = max(0, x - margin)
    x2 = min(image_rgb.shape[1], x + w + margin)
    
    cropped_object = image_rgb[y1:y2, x1:x2]
    
    # Valutazione CLIP
    inputs_image = clip_processor(images=cropped_object, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs_image)
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    similarity = (text_features @ image_features.T).item()
    
    # Print di debug per vedere chi sta vincendo
    # print(f"Oggetto {i} (Area: {mask_area}) - Score: {similarity:.3f}")
    
    if similarity > highest_score:
        highest_score = similarity
        best_mask = ann['segmentation']

# 7. RISULTATO FINALE
print(f"3/3 -> Trovato! Testati {valid_objects_tested} oggetti validi.")
print(f"🏆 Punteggio di confidenza del vincitore: {highest_score:.3f}")

if best_mask is None:
    print("Nessun oggetto trovato che rispetti i filtri di grandezza!")
else:
    darkened_image = (image_rgb * 0.3).astype(np.uint8)
    highlight_color = np.array([50, 255, 50]) # Messo verde fluo per vederlo meglio
    
    final_image = darkened_image.copy()
    final_image[best_mask] = (image_rgb[best_mask] * 0.6 + highlight_color * 0.4).astype(np.uint8)
    
    plt.figure(figsize=(10,10))
    plt.imshow(final_image)
    plt.axis('off')
    plt.title(f"Risultato per: '{TARGET_OBJECT}'")
    plt.show()

# ==========================================
# FASE 4: LA MAGIA DELLA COMPUTER VISION (2D -> 3D)
# ==========================================
print("\n" + "="*40)
print("🌐 AVVIO PROIEZIONE 3D (Structure from Motion)")
print("="*40)

COLMAP_TXT_DIR = "./model" # <--- CONTROLLA CHE SIA LA CARTELLA GIUSTA

# Funzione matematica per convertire Quaternioni in Matrice di Rotazione (R)
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1]**2 - 2 * qvec[3]**2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

print("1/4 -> Lettura dei punti 3D di COLMAP...")
points3D_path = os.path.join(COLMAP_TXT_DIR, "points3D.txt")
xyz = []
rgb = []
point_ids = []
with open(points3D_path, "r") as f:
    for line in f:
        if line.startswith("#"): continue
        elems = line.split()
        point_ids.append(int(elems[0]))
        xyz.append([float(elems[1]), float(elems[2]), float(elems[3])])
        rgb.append([float(elems[4])/255.0, float(elems[5])/255.0, float(elems[6])/255.0])
xyz = np.array(xyz)
rgb = np.array(rgb)

print("2/4 -> Estrazione parametri della telecamera (Intrinsics & Extrinsics)...")
# Troviamo la telecamera che ha scattato la foto che stiamo analizzando
target_image_name = os.path.basename(IMAGE_PATH)
R, t, camera_id = None, None, None

with open(os.path.join(COLMAP_TXT_DIR, "images.txt"), "r") as f:
    lines = f.readlines()
    for i in range(0, len(lines), 2):
        if lines[i].startswith("#"): continue
        elems = lines[i].split()
        name = elems[9]
        if name == target_image_name:
            qvec = np.array(tuple(map(float, elems[1:5])))
            t = np.array(tuple(map(float, elems[5:8])))
            R = qvec2rotmat(qvec)
            camera_id = elems[8]
            break

if R is None:
    raise ValueError(f"Immagine {target_image_name} non trovata nei file di COLMAP!")

# Leggiamo gli Intrinsics (Matrice K)
K = np.eye(3)
with open(os.path.join(COLMAP_TXT_DIR, "cameras.txt"), "r") as f:
    for line in f:
        if line.startswith("#"): continue
        elems = line.split()
        if elems[0] == camera_id:
            # Assumiamo modello SIMPLE_RADIAL (f, cx, cy, k) o PINHOLE
            focal_length = float(elems[4])
            cx = float(elems[5])
            cy = float(elems[6])
            K[0,0] = focal_length
            K[1,1] = focal_length
            K[0,2] = cx
            K[1,2] = cy
            break

print("3/4 -> Esecuzione proiezione matematica (x = K[R|t]X)...")
# Trasformiamo i punti 3D dal sistema "mondo" al sistema "telecamera"
xyz_cam = (R @ xyz.T).T + t

# Filtriamo i punti che sono fisicamente dietro la telecamera (Z < 0)
valid_depth = xyz_cam[:, 2] > 0
xyz_cam_valid = xyz_cam[valid_depth]
original_indices = np.where(valid_depth)[0]

# Proiettiamo i punti sul sensore 2D (divisione prospettica)
uv_homog = (K @ xyz_cam_valid.T).T
u = (uv_homog[:, 0] / uv_homog[:, 2]).astype(int)
v = (uv_homog[:, 1] / uv_homog[:, 2]).astype(int)

print("4/4 -> Incrocio spaziale: quali punti 3D cadono nella maschera AI?")
# Controlliamo quali coordinate 2D proiettate cadono esattamente DENTRO la best_mask di SAM
height, width = best_mask.shape
object_3d_indices = []

for idx, (px_u, px_v) in enumerate(zip(u, v)):
    # Controlla che il pixel sia dentro i bordi dell'immagine
    if 0 <= px_u < width and 0 <= px_v < height:
        # Se in quel pixel la maschera è True (appartiene alla borraccia)
        if best_mask[px_v, px_u]: 
            # Salva l'indice del punto 3D originale
            object_3d_indices.append(original_indices[idx])

print(f"🎯 BINGO! Trovati {len(object_3d_indices)} punti 3D che appartengono all'oggetto cercato.")

# ==========================================
# VISUALIZZAZIONE 3D CON OPEN3D
# ==========================================
# Coloriamo di rosso acceso i punti trovati
for idx in object_3d_indices:
    rgb[idx] = [1.0, 0.0, 0.0]  # Rosso puro RGB

# Creiamo l'oggetto PointCloud per Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

# Aumentiamo la dimensione dei punti per vederli meglio
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultUnlit"
mat.point_size = 5.0 

print("\n🚀 Apertura del visualizzatore 3D. Usa il mouse per ruotare la scena!")
o3d.visualization.draw_geometries([pcd])
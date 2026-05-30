# SemanticSpot 3D

**Team:** Andrea Migliore · Giovanni Elisei · Carlo Fiammenghi

## Summary

SemanticSpot 3D is an end-to-end pipeline that turns a **walkthrough video** of a room into a **semantically searchable 3D point cloud**.  
A user can type a natural-language query (e.g. *"silver water bottle"*) and the system highlights the matching object directly inside a 3D viewer.

### How it works

| Stage | What happens | Tools |
|---|---|---|
| **1. Data Capture** | Record a video walking around the environment | Phone camera |
| **2. Structure from Motion** | Extract frames → estimate camera poses → reconstruct a dense 3D point cloud | COLMAP |
| **3. Semantic Projection** | Segment every frame with SAM, embed each segment with CLIP, then project the features onto the 3D points via known camera intrinsics/extrinsics | SAM (ViT-B) + CLIP (ViT-B/32) |
| **4. Semantic Search** | At query time, encode the text with CLIP, compute cosine similarity against every point's feature vector, and highlight the best matches | CLIP + DBSCAN (spatial cleanup) |

## Demo Video

https://youtu.be/KG0jpMZ2JLM

The Streamlit app allows you to:

- **Search by text**: type an object description to highlight matching points in the 3D model.
- **Visualize consistency**: show a heatmap based on how often points were detected across different camera angles.
- **Filter noise**: run DBSCAN to remove isolated points and outliers.
- **Adjust settings**: change similarity thresholds, consistency limits, and point counts in real time.

---

## Run the Demo (pre-computed scene)

Run the app on the pre-built scene.

### Prerequisites

- **Python 3.10+**
- **Git**

### 1. Clone and install

```bash
git clone https://github.com/AndreMigX/SemanticSpot3D.git
cd SemanticSpot3D

python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

### 2. Download the pre-computed scene

The preprocessed semantic scene file (`semantic_scene.npz`, ~637 MB) is too large for Git. Download it into the project root:

```bash
curl -L -o semantic_scene.npz \
  "https://docs.google.com/uc?export=download&id=1nosa5p3XwNpPgQvJaOqPbCTfPBMOMP-E&confirm=t"
```

### 3. Launch the app

```bash
streamlit run app.py
```

Or use the scripts:

```bash
./run.sh          # macOS / Linux
run.bat           # Windows
```

The app will open automatically in your browser at `http://localhost:8501`.

## Run the Full Pipeline from Scratch

Process a **new video** and generate your own `semantic_scene.npz`, then launch the demo on it.

### Prerequisites

- Everything from above
- [COLMAP](https://colmap.github.io/) installed on your system
- A walkthrough video of the scene you want to reconstruct

### Step 1: Extract frames from the video

Place your video as `src/video.mp4`, then run:

```bash
python src/extract_frames.py
```

This extracts frames at 3 fps into `src/frames/`.

### Step 2: Run COLMAP

Use COLMAP to reconstruct the 3D scene from the extracted frames:

1. Import the frames and run feature extraction + matching
2. Run sparse reconstruction, then dense reconstruction (patch-match stereo + fusion)
3. Export the model in **text format** (`cameras.txt`, `images.txt`, `points3D.txt`) into `colmap/model/`
4. Place the fused dense point cloud (`fused.ply`) in `colmap/`

> **Note:** Extracting the dense point cloud in COLMAP requires an NVIDIA GPU with CUDA.

### Step 3: Download the SAM checkpoint

The SAM ViT-B checkpoint (~358 MB) is required for segmentation:

```bash
curl -L -o sam_vit_b_01ec64.pth \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
```

### Step 4: Run the preprocessing notebook

Open and run all cells in `preprocessor.ipynb`. This will:

1. Load the dense point cloud and voxel-downsample it
2. For each video frame: run SAM segmentation → CLIP embedding → 3D projection
3. Average the CLIP features across views and compute consistency scores
4. Save everything to `semantic_scene.npz`

### Step 5: Launch the app

```bash
streamlit run app.py
```

The app loads `semantic_scene.npz` from the project root

## Project Structure

```
SemanticSpot3D/
├── app.py                  # Streamlit app
├── preprocessor.ipynb      # Pipeline to create semantic point cloud
├── requirements.txt        # Python dependencies
├── run.sh / run.bat        # Launch scripts
├── src/
│   ├── extract_frames.py   # Frames extraction utility
│   ├── frames/             # Extracted video frames
│   └── video.mp4           # Input video
├── colmap/
│   ├── fused.ply           # Dense 3D point cloud from COLMAP
│   └── model/              # COLMAP text exports (cameras, images, points3D)
├── semantic_scene.npz      # Semantic scene (gitignored, download separately)
└── sam_vit_b_01ec64.pth    # SAM checkpoint (gitignored, download separately)
```

## Tech Stack

| Component | Technology |
|---|---|
| 3D Reconstruction | [COLMAP](https://colmap.github.io/) |
| Image Segmentation | [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) - ViT-B |
| Semantic Features | [CLIP](https://github.com/openai/CLIP) - ViT-B/32 (HuggingFace Transformers) |
| Spatial Filtering | DBSCAN (scikit-learn) |
| App | [Streamlit](https://streamlit.io/) + [Plotly](https://plotly.com/) |
| Point Cloud I/O | [Open3D](http://www.open3d.org/) |

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import torch
import plotly.graph_objects as go
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="SemanticSpot 3D", layout="wide")

st.title("SemanticSpot 3D")


# Resource loading and caching

@st.cache_resource
def load_clip():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

@st.cache_data
def load_scene():
    data = np.load("semantic_scene.npz")
    xyz = data['xyz']
    rgb = data['rgb']
    features = data['features']
    consistency = data['consistency'] if 'consistency' in data else np.ones(len(xyz))
    return xyz, rgb, features, consistency

@st.cache_data
def encode_query(query: str, _model, _processor, device: str) -> np.ndarray:
    prompt = f"a photo of a {query}"
    inputs = _processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = _model.get_text_features(**inputs)
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]


with st.spinner("Loading model and scene data..."):
    clip_model, clip_processor, device = load_clip()
    xyz, rgb, features, consistency = load_scene()


# Sidebar controls

st.sidebar.header("Search")
search_query = st.sidebar.text_input("Search for an object:", placeholder="e.g. silver water bottle")
threshold = st.sidebar.slider("CLIP confidence threshold", min_value=0.20, max_value=0.35, value=0.28, step=0.01)

st.sidebar.markdown("---")
st.sidebar.header("Visualization")
show_consistency_heatmap = st.sidebar.checkbox("Show consistency heatmap", value=False)
min_consistency = st.sidebar.slider("Min consistency (removes shadows)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("Cleanup")
max_points = st.sidebar.slider("Max points (top-K)", min_value=500, max_value=50000, value=10000, step=500)
use_clustering = st.sidebar.checkbox("Enable spatial filter (DBSCAN)", value=True)
cluster_sensitivity = st.sidebar.slider("Filter sensitivity", min_value=0.01, max_value=0.20, value=0.05, step=0.01)


# Set point colors and sizes
if show_consistency_heatmap:
    cmap = plt.get_cmap('plasma')
    colors = (cmap(consistency)[:, :3] * 255).astype(np.uint8)
else:
    colors = (rgb * 255).astype(np.uint8)

sizes = np.full(len(xyz), 2)

# Semantic search logic

if search_query:
    text_vec = encode_query(search_query, clip_model, clip_processor, device)
    sims = features @ text_vec

    mask = (sims > threshold) & (consistency >= min_consistency)

    if np.any(mask):
        # Apply Top-K filtering
        if np.sum(mask) > max_points:
            cutoff = np.sort(sims[mask])[-max_points]
            mask = mask & (sims >= cutoff)

        # Apply spatial filtering (DBSCAN) to remove outliers
        if use_clustering:
            pts = xyz[mask]
            if len(pts) > 10:
                labels = DBSCAN(eps=cluster_sensitivity, min_samples=5).fit(pts).labels_
                noise = labels == -1
                valid_idx = np.where(mask)[0]
                mask[valid_idx[noise]] = False
                st.sidebar.success(f"DBSCAN removed {int(noise.sum())} outliers")

    if np.any(mask):
        if not show_consistency_heatmap:
            # Color non-matching points gray and highlight search matches in green
            colors[~mask] = [40, 40, 40]
            colors[mask] = [50, 255, 50]
        else:
            # Attenuate non-matching points to emphasize the consistency heatmap
            colors[~mask] = (colors[~mask] * 0.2).astype(np.uint8)

        sizes[mask] = 5
        st.success(f"Found {int(np.sum(mask))} matching points.")
    else:
        st.warning("Nothing found — try lowering the threshold.")


# Render using a custom HTML component to persist the 3D camera angle across Streamlit re-runs.

if show_consistency_heatmap:
    hover_text = [f"Consistency: {c:.2f}" for c in consistency]
else:
    hover_text = [f"RGB: {r},{g},{b}" for r, g, b in colors]

fig = go.Figure(data=[go.Scatter3d(
    x=xyz[:, 0],
    y=xyz[:, 1],
    z=xyz[:, 2],
    mode='markers',
    hoverinfo='text',
    text=hover_text,
    marker=dict(
        size=sizes,
        color=[f'rgb({r},{g},{b})' for r, g, b in colors],
        opacity=0.9
    )
)])

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor='black'
    ),
    margin=dict(r=0, l=0, b=0, t=0),
    height=700,
    paper_bgcolor='black'
)

fig_json = fig.to_json()

# Embed Plotly inside an HTML iframe and store the camera state in window.parent to preserve orientation during Streamlit updates.
html_plot = f"""
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; background: black; }}
    #plot {{ width: 100%; height: 710px; }}
  </style>
</head>
<body>
<div id="plot"></div>
<script>
  const figData = {fig_json};
  const saved = window.parent._plotCamera3d;
  if (saved) {{
    figData.layout.scene = figData.layout.scene || {{}};
    figData.layout.scene.camera = saved;
  }}

  Plotly.react('plot', figData.data, figData.layout, {{responsive: true}})
    .then(function(gd) {{
      gd.on('plotly_relayout', function(ed) {{
        if (ed && ed['scene.camera']) {{
          window.parent._plotCamera3d = ed['scene.camera'];
        }}
      }});
    }});
</script>
</body>
</html>
"""

components.html(html_plot, height=720)
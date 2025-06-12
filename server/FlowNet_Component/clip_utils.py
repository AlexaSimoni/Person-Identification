from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2
import torch
import numpy as np
import logging

from server.Utils.framesGlobals import flow_clip_reference, all_even_frames

logger = logging.getLogger(__name__)

# Load the pre-trained CLIP model and processor from HuggingFace
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Converts a person crop to a CLIP embedding for visual similarity comparison of full-body crops
# Input: image in BGR format (NumPy array from OpenCV)
# Output: CLIP embedding (L2-normalized NumPy array)
def get_clip_embedding(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB
    pil_image = Image.fromarray(image_rgb)  # Convert to PIL image
    inputs = clip_processor(images=pil_image, return_tensors="pt")  # Preprocess for CLIP
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)  # Extract feature vector by CLIP image encoder
    features = features / features.norm(dim=-1, keepdim=True)  # Normalize to unit vector (L2)
    return features[0].cpu().numpy()  # Return as NumPy array

# Measures how visually similar two crops are
# Input: two CLIP embeddings (NumPy arrays)
# Output: cosine similarity value between -1.0 and 1.0
def compare_clip_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
    emb1 = np.array(emb1, dtype=np.float32)
    emb2 = np.array(emb2, dtype=np.float32)
    return float(np.dot(emb1, emb2))  # Cosine similarity normalized

# Add a CLIP reference embedding to the person’s list of known appearances (max 5 recent)
# Input: UUID string, CLIP embedding vector, optional similarity score
# Output: None (updates global dictionary flow_clip_reference)
def add_clip_reference(uuid: str, clip_embedding, similarity: float = -1, max_refs: int = 5):
    if uuid not in flow_clip_reference:
        flow_clip_reference[uuid] = {
            "clip_embeddings": [],
            "similarities": []
        }
    flow_clip_reference[uuid]["clip_embeddings"].append(clip_embedding)
    flow_clip_reference[uuid]["similarities"].append(similarity)

    # Keep only the most recent max_refs embeddings
    if len(flow_clip_reference[uuid]["clip_embeddings"]) > max_refs:
        flow_clip_reference[uuid]["clip_embeddings"].pop(0)
        flow_clip_reference[uuid]["similarities"].pop(0)
    logger.info(f"[CLIP] Added reference for UUID {uuid} (total: {len(flow_clip_reference[uuid]['clip_embeddings'])})")

# Determine whether to save a fallback or not
# Input: UUID string
# Output: True - person has stored CLIP references, False - otherwise
def has_clip_reference(uuid: str) -> bool:
    return uuid in flow_clip_reference and bool(flow_clip_reference[uuid].get("clip_embeddings"))

# CLIP reference for similarity checking against new frames
# Input: UUID string
# Output: list of stored CLIP embeddings for that person (as float32 NumPy arrays)
def get_clip_reference(uuid: str):
    refs = flow_clip_reference.get(uuid, {}).get("clip_embeddings", [])
    return [np.array(ref, dtype=np.float32) for ref in refs]

# Check if unique to prevent saving nearly identical crops
# Input: UUID string, new CLIP embedding, similarity threshold (default 0.05)
# Output: True - new embedding is visually different enough from recent ones, False - otherwise
def is_unique_against_recent_clip_refs(uuid: str, new_emb: np.ndarray, threshold: float) -> bool:
    refs = flow_clip_reference.get(uuid, {}).get("clip_embeddings", [])
    recent_refs = refs[-3:]  # Only compare to last 3 saved references
    for ref in recent_refs:
        sim = compare_clip_embeddings(new_emb, ref)
        if abs(sim - 1.0) < threshold:
            logger.info(f"[CLIP] Skipping frame — too similar to recent (sim={sim:.4f})")
            return False
    return True

# Called once after the first FaceNet match
# captures a fallback appearance for tracking
# Input: UUID string, frame index (int), bounding box (x, y, w, h)
# Output: None (saves first CLIP embedding into memory)
def try_save_initial_clip_reference(uuid: str, frame_index: int, box: tuple):
    # Skip if already initialized
    if uuid in flow_clip_reference and flow_clip_reference[uuid].get("clip_embeddings"):
        return

    frame = all_even_frames.get(frame_index)  # get frame by given frame index from memory
    if frame is None:   # The frame isn't available
        logger.warning(f"[CLIP] No frame found at index {frame_index} for UUID {uuid}")
        return

    x, y, w, h = box
    crop = frame[y:y + h, x:x + w]  # Crop the person region
    clip_emb = get_clip_embedding(crop)
    if clip_emb is not None:  # If the embedding was successfully created add it as a reference for this UUID
        add_clip_reference(uuid, clip_emb, similarity=-1)
        logger.info(f"[CLIP] Saved initial fallback reference for UUID {uuid} at frame {frame_index}")

# Stores fallback appearance (if FaceNet fails) for FlowNet identify validation
# Input: UUID string, frame index (int), bounding box (x, y, w, h), similarity (float)
# Output: None (saves CLIP embedding)
def save_clip_reference_on_low_similarity(uuid: str, frame_index: int, box: tuple, similarity: float):
    frame = all_even_frames.get(frame_index) # get frame by given frame index from memory

    if frame is None:   # The frame isn't available
        logger.warning(f"[CLIP] No frame found at index {frame_index} for UUID {uuid}")
        return

    x, y, w, h = box
    crop = frame[y:y + h, x:x + w]  # Crop the person region
    clip_emb = get_clip_embedding(crop)
    if clip_emb is not None:    # If the embedding was successfully created add it as a reference for this UUID
        add_clip_reference(uuid, clip_emb, similarity=similarity)
        logger.info(f"[CLIP] Saved low-similarity fallback reference for UUID {uuid} at frame {frame_index}")

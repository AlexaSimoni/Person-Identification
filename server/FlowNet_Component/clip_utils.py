from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2
import torch
import numpy as np
import logging

from server.Utils.framesGlobals import flow_clip_reference
logger = logging.getLogger(__name__)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_embedding(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    inputs = clip_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy()

def compare_clip_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
    emb1 = np.array(emb1, dtype=np.float32)
    emb2 = np.array(emb2, dtype=np.float32)
    return float(np.dot(emb1, emb2))  # Cosine similarity (CLIP outputs are normalized)

def add_clip_reference(uuid: str, clip_embedding, similarity: float = -1, max_refs: int = 5):
    #Add a CLIP reference embedding to the global flow_clip_reference dictionary.
    if uuid not in flow_clip_reference:
        flow_clip_reference[uuid] = {
            "clip_embeddings": [],
            "similarities": []
        }

    flow_clip_reference[uuid]["clip_embeddings"].append(clip_embedding)
    flow_clip_reference[uuid]["similarities"].append(similarity)

    if len(flow_clip_reference[uuid]["clip_embeddings"]) > max_refs:
        flow_clip_reference[uuid]["clip_embeddings"].pop(0)
        flow_clip_reference[uuid]["similarities"].pop(0)

    logger.info(f"[CLIP] Added reference for UUID {uuid} (total: {len(flow_clip_reference[uuid]['clip_embeddings'])})")

def get_best_clip_similarity(uuid: str, new_embedding: np.ndarray) -> float:
    refs = flow_clip_reference.get(uuid, {}).get("clip_embeddings", [])
    if not refs:
        return 0.0
    similarities = [compare_clip_embeddings(new_embedding, ref) for ref in refs]
    return max(similarities)

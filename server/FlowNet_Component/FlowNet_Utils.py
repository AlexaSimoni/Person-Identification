import logging
import threading
import time
import base64
import io
from PIL import Image
import cv2
import numpy as np
from typing import Union
from server.FlowNet_Component.TrackingManager import TrackingManager
from server.Utils.framesGlobals import all_even_frames, flowdetected_frames
from server.config.config import USE_FLOWNETS, SIMILARITY_THRESHOLD, USE_CLIP_IN_FLOWTRACKING
from server.FaceNet_Componenet.FaceNet_Utils import face_embedding, embedding_manager
#from server.FlowNet_Component.clip_utils import get_clip_embedding, compare_clip_embeddings, add_clip_reference, get_clip_reference
from server.Utils.db import detected_frames_collection
if USE_CLIP_IN_FLOWTRACKING:
    from server.FlowNet_Component.clip_utils import (
        get_clip_embedding,
        compare_clip_embeddings,
        add_clip_reference,
        get_clip_reference,
        is_unique_against_recent_clip_refs
    )
logger = logging.getLogger(__name__)
logger.info(f"Using {'FlowNetS' if USE_FLOWNETS else 'SimpleFlowNet (Farneback)'} for optical flow.")

# FlowNet_Utils manages the main FlowNet tracking logic, including the tracking loop,
# integration with CLIP and FaceNet embeddings, and insertion of tracked crops into MongoDB
# It supports concurrent tracking via threading and uses both optical flow and image similarity to determine valid detections

# Singleton control to start FlowNet tracking loop only once
flow_tracking_started = False
flow_tracking_lock = threading.Lock()
tracking_manager = TrackingManager()


# Start the background FlowNet tracking loop once
def start_flow_tracking():
    global flow_tracking_started

    with flow_tracking_lock:
        if flow_tracking_started:
            return
        flow_tracking_started = True
        threading.Thread(target=flow_tracking_loop, daemon=True).start()
        logger.info("FlowNet tracking thread started.")

# Background loop that applies FlowNet tracking to new frames
# Continuously checks for the latest frame index and updates tracked boxes
def flow_tracking_loop():
    logger.info("FlowNet tracking loop running...")
    last_index = -1
    try:
        while True:
            if all_even_frames:
                latest_index = max(all_even_frames.keys())
                if latest_index > last_index:
                    last_index = latest_index
                    tracking_manager.update_all(latest_index)

            time.sleep(0.03)  #Sleep to avoid busy loop (CPU)
    except Exception as e:
        logger.exception(f"[FlowNet] Tracking thread crashed: {e}")

# Returns the singleton instance of the TrackingManager
def get_tracking_manager():
    return tracking_manager

# Draws FlowNet tracked boxes and labels on the frame during annotation
# inputs: frame (image array), frame_index
# output: frame modified with drawn rectangles
def draw_tracking_boxes(frame, frame_index):
    tracker_manager = get_tracking_manager()

    for tracker in tracker_manager.get_all():
        if tracker.last_box is not None:
            x, y, w, h = tracker.last_box
            label = f"FlowNet | Best: {tracker.best_score:.2f}%"
            logger.info(f"FlowNet box for {tracker.uuid} at frame {frame_index}: {tracker.last_box}")

            #Draw bounding box in orange
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)

            #Label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = x
            text_y = y - 10 if y - 10 > 10 else y + text_size[1] + 10

            cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5),
                          (text_x + text_size[0], text_y + 5), (0, 165, 255), cv2.FILLED)
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

# Inserts FlowNet-tracked frames into MongoDB with FaceNet and CLIP filtering
# Inputs: uuid (person id), running_id (session id of current video run), frame_per_second
# Output: None (inserts filtered frames and optionally new embeddings into the database)
async def insert_flow_detected_frames(uuid: str, running_id: str, frame_per_second: int = 30):
    #if USE_CLIP_IN_FLOWTRACKING:
        #from server.FlowNet_Component.FlowTracker import is_unique_against_recent_clip_refs

    inserted_count = 0
    unique_embeddings = []
    CLIP_SIMILARITY_THRESHOLD = 0.75    # Minimum similarity for CLIP fallback acceptance
    CLIP_UPDATE_THRESHOLD = 0.85    # Similarity required to update CLIP references

    # Load reference FaceNet embeddings for this person
    reference_record = await get_reference_record(uuid)
    if reference_record is None:
        return

    # Confirm FlowNet tracker running for this person
    tracker = tracking_manager.trackers.get(uuid)
    if not tracker:
        logger.warning(f"[FlowNet] No tracker found for UUID {uuid}")
        return

    # Iterate over all frames collected by FlowNet
    for frame_index, frame_data in flowdetected_frames.items():
        base64_crop = frame_data.get("cropped_image")
        if not base64_crop:
            logger.warning(f"[FlowNet] No image found in flowdetected_frames[{frame_index}]")
            continue

        # Decode image from base64
        image = decode_base64_to_image(base64_crop)
        if image is None:
            continue

        # Compute FaceNet embedding and similarity
        embedding = compute_embedding(image)
        similarity = 0.0
        embedded = False
        is_unique = False
        if embedding is not None:
            similarity = compute_similarity(reference_record, base64_crop)

        # If FaceNet fails or similarity is low, run CLIP fallback
        #if embedding is None or similarity < SIMILARITY_THRESHOLD:

        # If FaceNet fails or similarity is low, optionally run CLIP fallback
        if embedding is None or similarity < SIMILARITY_THRESHOLD:
            if USE_CLIP_IN_FLOWTRACKING:

                clip_emb = get_clip_embedding(np.array(image.convert("RGB")))
                if clip_emb is None or not clip_emb.any():
                    logger.warning(f"[FlowNet] No CLIP embedding for frame {frame_index}")
                    continue

                # Compare to existing CLIP references
                ref_embs = get_clip_reference(uuid)
                similarities = [compare_clip_embeddings(clip_emb, ref) for ref in ref_embs]
                max_sim = max(similarities, default=0.0)
                logger.info(f"[CLIP] Similarity to reference for UUID {uuid} at frame {frame_index}: {max_sim:.2f}")
                # If below minimum similarity, reject the frame
                if max_sim < CLIP_SIMILARITY_THRESHOLD:
                    logger.warning(f"[FlowNet] CLIP fallback failed — skipping frame {frame_index}")
                    continue
                # If high similarity, update CLIP references
                if max_sim >= CLIP_UPDATE_THRESHOLD:
                    add_clip_reference(uuid, clip_emb, similarity=max_sim)
                    logger.info(f"[CLIP] Updated CLIP references for UUID {uuid} with sim={max_sim:.2f}")
                # Prevent saving near-duplicate CLIP embeddings
                if not is_unique_against_recent_clip_refs(uuid, clip_emb, threshold=0.05):
                    logger.info(f"[FlowNet] CLIP fallback duplicate — skipping frame {frame_index}")
                    continue

                logger.info(f"[FlowNet] CLIP verified and unique — accepting fallback frame {frame_index}")

                """
                if not clip_filter_passes(uuid, image, threshold=CLIP_SIMILARITY_THRESHOLD):
                    logger.warning(f"[FlowNet] CLIP fallback failed — skipping frame {frame_index}")
                    continue
                else:
                    logger.info(f"[FlowNet] CLIP verified — allowing fallback frame {frame_index}")
                """
            else:
                # CLIP disabled and FaceNet similarity too low — skip frame
                logger.warning(f"[FlowNet] Frame {frame_index} below FaceNet threshold and CLIP disabled — skipping")
                continue

        # If FaceNet embedding exists, check if it's unique
        if embedding is not None:
            embedded = True
            #is_unique = await is_unique_flownet_embedding(uuid, embedding)
            is_unique = await is_unique_flownet_embedding(reference_record, embedding)

            if is_unique:
                logger.info(f"[FlowNet] Frame {frame_index} embedding is UNIQUE")
                unique_embeddings.append(embedding)

        # Insert final document to MongoDB
        document = build_flownet_document(
            uuid, running_id, frame_index, base64_crop,
            similarity, frame_per_second, embedded, is_unique
        )
        await detected_frames_collection.insert_one(document)
        inserted_count += 1

    # Save any new embeddings
    if unique_embeddings:
        logger.info(f"[FlowNet] Storing {len(unique_embeddings)} new unique FlowNet embeddings to DB.")
        await embedding_manager.save_embeddings_to_db(uuid, unique_embeddings)
        logger.info(f"[FlowNet] Saved {len(unique_embeddings)} new FlowNet embeddings to DB")

    logger.info(f"[FlowNet] Inserted {inserted_count} FlowNet-tracked frames to MongoDB.")

# Fetches FaceNet reference embeddings for a person
# Inputs: uuid (person id)
# Output: record from DB or None
async def get_reference_record(uuid: str):
    record = await embedding_manager.get_reference_embeddings(uuid)
    if record is None or "embeddings" not in record:
        logger.warning(f"[FlowNet] No reference embeddings found for UUID {uuid}")
        return None
    return record

# Converts a base64 string into a image
# Inputs: base64_crop (str)
# Output: PIL Image object or None
def decode_base64_to_image(base64_crop: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(base64.b64decode(base64_crop)))
    except Exception as e:
        logger.error(f"[FlowNet] Failed to decode image: {e}")
        return None

# Computes FaceNet embedding for image
# Inputs: image (PIL Image)
# Output: numpy ndarray embedding
def compute_embedding(image: Image.Image) -> Union[np.ndarray, None]:
    embedding = face_embedding.get_embedding(image)
    return embedding

# Computes similarity between given crop and reference embeddings
# Inputs: reference_record (DB object), base64_crop (str)
# Output: similarity score (float)
def compute_similarity(reference_record, base64_crop: str) -> float:
    return embedding_manager.calculate_similarity(reference_record, base64_crop)

# Creates a document for inserting tracked frame data to DB
# Inputs: uuid (person id), running_id (session id of current video run), frame_index, base64_crop (cropped), similarity,
#         fps (frame per second), embedded (flag), is_unique (flag)
# Output: dict formatted document for MongoDB
def build_flownet_document(uuid: str, running_id: str, frame_index: int, base64_crop: str,
                           similarity: float, fps: int, embedded: bool, is_unique: bool):
    return {
        "uuid": uuid,
        "running_id": running_id,
        "frame_index": frame_index,
        "frame_data": {
            "cropped_image": base64_crop,
            "similarity": similarity
        },
        "embedded": True,
        "is_unique": is_unique,
        "frame_per_second": fps,
        "source": "flownet"
    }
"""
# Checks if the new FaceNet embedding is unique enough to store
# inputs: uuid (person id), new_embedding (np.ndarray), threshold (float cosine distance)
# output: True if unique, False otherwise
async def is_unique_flownet_embedding(uuid: str, new_embedding: np.ndarray, threshold: float = 0.2) -> bool:
    # Retrieve the stored reference embeddings from the database for the given person UUID
    reference_record = await embedding_manager.get_reference_embeddings(uuid)
    if not reference_record or "embeddings" not in reference_record:
        return True  # No reference embeddings -> unique

    # Convert each stored embedding from the reference record into a NumPy array for comparison
    existing_embeddings = [np.array(e) for e in reference_record.get("embeddings", [])]
    # Check if the new embedding is different enough from existing ones
    return embedding_manager.is_unique_embedding(new_embedding, existing_embeddings, threshold)

"""
    # Checks if the new FaceNet embedding is unique enough to store
    # inputs: reference_record (dict from DB with stored embeddings for this person),
    #       new_embedding (FaceNet embedding), threshold (max cosine similarity for unique)
    # output: True - unique, False - otherwise
async def is_unique_flownet_embedding(reference_record: dict, new_embedding: np.ndarray, threshold: float = 0.02) -> bool:
    # If no stored embeddings - treat as unique
    if not reference_record or "embeddings" not in reference_record:
        return True
    # Convert each stored embedding to NumPy array for comparison
    existing_embeddings = [np.array(e) for e in reference_record.get("embeddings", [])]
    # Return whether the new embedding is unique or existing
    return embedding_manager.is_unique_embedding(new_embedding, existing_embeddings, threshold)


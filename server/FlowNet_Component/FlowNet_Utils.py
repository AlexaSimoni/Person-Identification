import logging
import threading
import time

import cv2
import base64
import io
from PIL import Image
import cv2
import numpy as np
from typing import Union
import asyncio
from server.FlowNet_Component.TrackingManager import TrackingManager
from server.Utils.framesGlobals import all_even_frames
from server.config.config import USE_FLOWNETS
from server.Utils.framesGlobals import flowdetected_frames
from server.Utils.db import detected_frames_collection
from server.FaceNet_Componenet.FaceNet_Utils import face_embedding, embedding_manager



from server.Utils.db import detected_frames_collection
from server.FaceNet_Componenet.FaceNet_Utils import face_embedding
from server.Utils import framesGlobals

logger = logging.getLogger(__name__)
logger.info(f"Using {'FlowNetS' if USE_FLOWNETS else 'SimpleFlowNet (Farneback)'} for optical flow.")

#Singleton control to start tracking only once
flow_tracking_started = False
flow_tracking_lock = threading.Lock()

tracking_manager = TrackingManager()

def start_flow_tracking():
    #Start the background FlowNet tracking loop once
    global flow_tracking_started

    with flow_tracking_lock:
        if flow_tracking_started:
            return
        flow_tracking_started = True
        threading.Thread(target=flow_tracking_loop, daemon=True).start()
        logger.info("FlowNet tracking thread started.")

def flow_tracking_loop():
    #Background loop that applies FlowNet tracking to new frames
    #Continuously checks for the latest frame index and updates tracked boxes
    logger.info("FlowNet tracking loop running...")
    last_index = -1

    while True:
        if all_even_frames:
            latest_index = max(all_even_frames.keys())
            if latest_index > last_index:
                last_index = latest_index
                tracking_manager.update_all(latest_index)

        time.sleep(0.03)  #Sleep to avoid busy loop (CPU)

def get_tracking_manager():
    return tracking_manager

def draw_tracking_boxes(frame, frame_index):
    #Draw tracked bounding boxes from FlowNet on the given frame
    #Used during annotation of even frames only
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
"""

async def insert_flowdetected_frames(uuid: str, running_id: str, frame_per_second: int = 30):
    
    #Insert all FlowNet-tracked detections stored in memory into MongoDB.
    
    for frame_index, frame_data in flowdetected_frames.items():
        document = {
            "uuid": uuid,
            "running_id": running_id,
            "frame_index": frame_index,
            "frame_data": frame_data,
            "embedded": False,
            "frame_per_second": frame_per_second
        }
        await detected_frames_collection.insert_one(document)

    logger.info(f"[FlowNet] Inserted {len(flowdetected_frames)} FlowNet-tracked frames to MongoDB.")
"""
async def insert_flowdetected_frames(uuid: str, running_id: str, frame_per_second: int = 30):
    inserted_count = 0
    unique_embeddings = []

    reference_record = await get_reference_record(uuid)
    if reference_record is None:
        return

    for frame_index, frame_data in flowdetected_frames.items():
        base64_crop = frame_data.get("cropped_image")
        if not base64_crop:
            logger.warning(f"[FlowNet] No image found in flowdetected_frames[{frame_index}]")
            continue

        image = decode_base64_to_image(base64_crop)
        if image is None:
            continue

        embedding = compute_embedding(image)
        if embedding is None:
            logger.warning(f"[FlowNet] No face detected at frame {frame_index}")
            similarity = 0.0
            embedded = False
            is_unique = False

        else:
            logger.info(f"[FlowNet] UUID {uuid} | Frame {frame_index} | Embedding preview: {embedding.flatten()[:5]}")
            similarity = compute_similarity(reference_record, base64_crop)
            embedded = True
            is_unique = await is_unique_flownet_embedding(uuid, embedding)

            if is_unique:
                logger.info(f"[FlowNet] Frame {frame_index} embedding is UNIQUE")
                unique_embeddings.append(embedding)

            else:
                logger.info(f"[FlowNet] Frame {frame_index} embedding is NOT unique")

        #logger.info(f"[FlowNet] UUID {uuid} | Frame {frame_index} | Embedding preview: {embedding.flatten()[:5]}")

        #similarity = compute_similarity(reference_record, base64_crop)

        #document = build_flownet_document(uuid, running_id, frame_index, base64_crop, similarity, frame_per_second)
        #document = build_flownet_document(uuid, running_id, frame_index, base64_crop, similarity, frame_per_second, embedded)
        document = build_flownet_document(
            uuid, running_id, frame_index, base64_crop,
            similarity, frame_per_second, embedded, is_unique
        )
        await detected_frames_collection.insert_one(document)
        inserted_count += 1

    if unique_embeddings:
        logger.info(f"[FlowNet] Storing {len(unique_embeddings)} new unique FlowNet embeddings to DB.")
        await embedding_manager.save_embeddings_to_db(uuid, unique_embeddings)

    logger.info(f"[FlowNet] Inserted {inserted_count} FlowNet-tracked frames to MongoDB.")


async def get_reference_record(uuid: str):
    record = await embedding_manager.get_reference_embeddings(uuid)
    if record is None or "embeddings" not in record:
        logger.warning(f"[FlowNet] No reference embeddings found for UUID {uuid}")
        return None
    return record

def decode_base64_to_image(base64_crop: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(base64.b64decode(base64_crop)))
    except Exception as e:
        logger.error(f"[FlowNet] Failed to decode image: {e}")
        return None

def compute_embedding(image: Image.Image) -> Union[np.ndarray, None]:
    embedding = face_embedding.get_embedding(image)
    return embedding

def compute_similarity(reference_record, base64_crop: str) -> float:
    return embedding_manager.calculate_similarity(reference_record, base64_crop)

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

async def is_unique_flownet_embedding(uuid: str, new_embedding: np.ndarray, threshold: float = 0.2) -> bool:
    reference_record = await embedding_manager.get_reference_embeddings(uuid)
    if not reference_record or "embeddings" not in reference_record:
        return True  # No reference embeddings → treat as unique

    existing_embeddings = [np.array(e) for e in reference_record.get("embeddings", [])]
    #return face_embedding.is_unique_embedding(new_embedding, existing_embeddings, threshold)
    return embedding_manager.is_unique_embedding(new_embedding, existing_embeddings, threshold)

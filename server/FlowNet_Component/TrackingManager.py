import logging
import numpy as np
from server.FlowNet_Component.FlowTracker import FlowTracker
from server.FlowNet_Component.SimpleFlowNet import SimpleFlowNet
from server.FlowNet_Component.FlowNetSWrapper import FlowNetSWrapper
from server.config.config import USE_FLOWNETS, FLOWNET_MODEL_PATH
from server.FlowNet_Component.clip_utils import get_clip_embedding, add_clip_reference
from server.Utils.framesGlobals import all_even_frames, flow_clip_reference

logger = logging.getLogger(__name__)

class TrackingManager:
    def __init__(self):
        self.trackers = {}  #uuid -> FlowTracker
        self.clip_references = {}  # UUID -> {"clip_embeddings": [np.ndarray, ...]}

        #Load FlowNet model once based on config flag
        if USE_FLOWNETS:
            logger.info(f"[TrackingManager] Using FlowNetS model from {FLOWNET_MODEL_PATH}")
            self.flow_net = FlowNetSWrapper(checkpoint_path=FLOWNET_MODEL_PATH)
        else:
            logger.info("[TrackingManager] Using SimpleFlowNet (Farneback)")
            self.flow_net = SimpleFlowNet()

    def match_or_add(self, box, similarity, frame_index, uuid, SIMILARITY_THRESHOLD):
        #Register or update a person for FlowNet tracking
        #Only update tracker if similarity is above the threshold
        if uuid not in self.trackers:
            self.trackers[uuid] = FlowTracker(flow_net=self.flow_net, uuid=uuid)
            logger.info(f"[FlowNet] Tracker created for UUID {uuid}")

        tracker = self.trackers[uuid]
        # Only initialize ONCE from FaceNet
       # if tracker.last_box is None and similarity >= SIMILARITY_THRESHOLD:

        if similarity >= SIMILARITY_THRESHOLD:
            tracker.last_box = box
            tracker.initial_facenet_box = box
            tracker.last_frame_index =frame_index
            tracker.frames_since_last_match = 0

            if similarity > tracker.best_score:
               tracker.best_score = similarity
            logger.info(
                f"[FlowNet] Initialized tracker for UUID {uuid} at frame {tracker.last_frame_index} | sim: {similarity:.2f}%")
            self.try_save_initial_clip_reference(uuid, frame_index, box)

        # Optional debug: prevent further updates
        else:
            # similarity too low — consider saving fallback
            self.save_clip_reference_on_low_similarity(uuid, frame_index, box, similarity)

    def update_all(self, frame_index):
        #Update all tracked boxes using FlowNet
        for tracker in self.trackers.values():
            logger.info(f"[TrackingManager] Updating tracker {tracker.uuid} using {type(tracker.flow_net).__name__}")
            tracker.update_track_frame(frame_index)


    def get_all(self):
        #Return all tracker objects
        return self.trackers.values()
    """
    def get_frame_index_from_frame(self, frame):
        #Safely extract frame index
        return getattr(frame, 'frame_index', None)
    """
    def get_flow_net(self):
        #Return the currently active flow_net instance
        return self.flow_net

    def try_save_initial_clip_reference(self, uuid: str, frame_index: int, box: tuple):
        if uuid in flow_clip_reference and flow_clip_reference[uuid].get("clip_embeddings"):
            return  # Already exists

        frame = all_even_frames.get(frame_index)
        if frame is None:
            logger.warning(f"[CLIP] No frame found at index {frame_index} for UUID {uuid}")
            return

        x, y, w, h = box
        crop = frame[y:y + h, x:x + w]
        clip_emb = get_clip_embedding(crop)
        if clip_emb is not None:
            add_clip_reference(uuid, clip_emb, similarity=-1)
            logger.info(f"[CLIP] Saved initial fallback reference for UUID {uuid} at frame {frame_index}")

    def save_clip_reference_on_low_similarity(self, uuid: str, frame_index: int, box: tuple, similarity: float):
        frame = all_even_frames.get(frame_index)
        if frame is None:
            logger.warning(f"[CLIP] No frame found at index {frame_index} for UUID {uuid}")
            return

        x, y, w, h = box
        crop = frame[y:y + h, x:x + w]
        clip_emb = get_clip_embedding(crop)
        if clip_emb is not None:
            add_clip_reference(uuid, clip_emb, similarity=similarity)
            logger.info(f"[CLIP] Saved low-similarity fallback reference for UUID {uuid} at frame {frame_index}")

    def has_clip_reference(self, uuid: str) -> bool:
        return uuid in flow_clip_reference and bool(flow_clip_reference[uuid].get("clip_embeddings"))

    def get_clip_reference(self, uuid: str):
        refs = flow_clip_reference.get(uuid, {}).get("clip_embeddings", [])

        return [np.array(ref, dtype=np.float32) for ref in refs]




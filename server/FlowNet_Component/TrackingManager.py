import logging
from server.FlowNet_Component.FlowTracker import FlowTracker
from server.FlowNet_Component.SimpleFlowNet import SimpleFlowNet
from server.FlowNet_Component.FlowNetSWrapper import FlowNetSWrapper
from server.Utils.framesGlobals import all_even_frames
from server.config.config import USE_FLOWNETS, FLOWNET_MODEL_PATH, FLOWNET_MATCH_FROM_FACENET_EVERY_TIME
from server.FlowNet_Component.clip_utils import try_save_initial_clip_reference, save_clip_reference_on_low_similarity, \
    get_clip_embedding

logger = logging.getLogger(__name__)

# TrackingManager manages all FlowTrackers for individual persons identified by UUID
# Handles model selection, tracker initialization, and per-frame updates
# Inputs: None (uses global config for model choice)
# Output: Maintains state of FlowTrackers and associated CLIP embeddings
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

    # Register or update a person’s FlowTracker based on FaceNet detection result
    # Only update tracker if similarity is above the threshold
    # Inputs: box (bounding box), similarity (FaceNet similarity score), frame_index (current), uuid (person ID), SIMILARITY_THRESHOLD
    # Output: None (Updates or creates a FlowTracker instance)
    def match_or_add(self, box, similarity, frame_index, uuid, SIMILARITY_THRESHOLD):
        # Create tracker if it doesn’t exist
        if uuid not in self.trackers:
            self.trackers[uuid] = FlowTracker(
                flow_net=self.flow_net,
                uuid=uuid,
                #use_clip=USE_CLIP_IN_FLOWTRACKING  # Pass CLIP toggle to tracker
            )
            logger.info(f"[FlowNet] Tracker created for UUID {uuid}")
        tracker = self.trackers[uuid]
        # (Only initialize ONCE from FaceNet:)
        # if tracker.last_box is None and similarity >= SIMILARITY_THRESHOLD:

        # Update tracker only if similarity is above threshold
        if similarity >= SIMILARITY_THRESHOLD:
            if FLOWNET_MATCH_FROM_FACENET_EVERY_TIME or tracker.last_frame_index is None:
                tracker.last_box = box
                tracker.initial_facenet_box = box
                tracker.last_frame_index = frame_index
                tracker.frames_since_last_match = 0

                # Save best match score
                if similarity > tracker.best_score:
                    tracker.best_score = similarity
                logger.info(
                    f"[FlowNet] Initialized tracker for UUID {uuid} at frame {tracker.last_frame_index} | sim: {similarity:.2f}%")

                # Save initial CLIP from FaceNet if enabled
                #if tracker.use_clip:
                if tracker.use_clip and tracker.initial_clip_embedding is None:

                    #try_save_initial_clip_reference(uuid, frame_index, box)
                    # Also store first CLIP embedding in tracker
                    #frame = all_even_frames.get(frame_index)
                    #if tracker.initial_clip_embedding is None:
                    frame = all_even_frames.get(frame_index)
                    if frame is not None:
                            x, y, w, h = box
                            crop = frame[y:y + h, x:x + w]
                            tracker.initial_clip_embedding = get_clip_embedding(crop)
                            logger.info(f"[CLIP] Initial embedding stored for UUID {uuid}")


        # Optional debug: prevent further updates
        else:
            # If similarity is too low and CLIP is enabled, try to save fallback
            if tracker.use_clip:
                save_clip_reference_on_low_similarity(uuid, frame_index, box, similarity)

    # Updates all FlowTrackers using optical flow for a new frame
    # Inputs: frame_index (current)
    # Output: None (Each tracker updates its box internally)
    def update_all(self, frame_index):
        for tracker in self.trackers.values():
            logger.info(f"[TrackingManager] Updating tracker {tracker.uuid} using {type(tracker.flow_net).__name__}")
            tracker.update_track_frame(frame_index)

    # Returns all active tracker objects
    # Inputs: None
    # Output: List of FlowTracker instances
    def get_all(self):
        return self.trackers.values()

    # Returns the currently used FlowNet model (either FlowNetSWrapper or SimpleFlowNet)
    # Inputs: None
    # Output: flow_net instance
    def get_flow_net(self):
        return self.flow_net






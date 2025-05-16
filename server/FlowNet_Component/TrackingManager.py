import logging
from server.FlowNet_Component.FlowTracker import FlowTracker
from server.FlowNet_Component.SimpleFlowNet import SimpleFlowNet
from server.FlowNet_Component.FlowNetSWrapper import FlowNetSWrapper
from server.config.config import USE_FLOWNETS, FLOWNET_MODEL_PATH

logger = logging.getLogger(__name__)

class TrackingManager:
    def __init__(self):
        self.trackers = {}  #uuid -> FlowTracker

        #Load FlowNet model once based on config flag
        if USE_FLOWNETS:
            logger.info(f"[TrackingManager] Using FlowNetS model from {FLOWNET_MODEL_PATH}")
            self.flow_net = FlowNetSWrapper(checkpoint_path=FLOWNET_MODEL_PATH)
        else:
            logger.info("[TrackingManager] Using SimpleFlowNet (Farneback)")
            self.flow_net = SimpleFlowNet()

    def match_or_add(self, box, similarity, frame, uuid, SIMILARITY_THRESHOLD):
        #Register or update a person for FlowNet tracking
        #Only update tracker if similarity is above the threshold
        if uuid not in self.trackers:
            self.trackers[uuid] = FlowTracker(flow_net=self.flow_net, uuid=uuid)
            logger.info(f"[FlowNet] Tracker created for UUID {uuid}")

        tracker = self.trackers[uuid]

        if similarity >= SIMILARITY_THRESHOLD:
            frame_index = self.get_frame_index_from_frame(frame)

            #Update current tracker state
            tracker.last_box = box
            tracker.last_frame_index = frame_index

            if similarity > tracker.best_score:
                tracker.best_score = similarity

            logger.info(f"[FlowNet-IoU] Updated tracker for UUID {uuid} at frame {frame_index} | similarity: {similarity:.2f}%")

    def update_all(self, frame_index):
        #Update all tracked boxes using FlowNet
        for tracker in self.trackers.values():
            tracker.update_track_frame(frame_index)


    def get_all(self):
        #Return all tracker objects
        return self.trackers.values()

    def get_frame_index_from_frame(self, frame):
        #Safely extract frame index
        return getattr(frame, 'frame_index', None)

    def get_flow_net(self):
        #Return the currently active flow_net instance
        return self.flow_net

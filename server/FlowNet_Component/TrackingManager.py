import logging

from server.FlowNet_Component.FlowTracker import FlowTracker
from server.config.config import SIMILARITY_THRESHOLD


#from server.FlowNet_Component.TrackedPerson import TrackedPerson


class TrackingManager:
    def __init__(self):
        self.trackers = {}  # uuid -> FlowTracker

    def match_or_add(self, box, similarity, frame, uuid, SIMILARITY_THRESHOLD):
        """
        Register or update a person for FlowNet tracking.
        Only update tracker if similarity is above the threshold.
        Stores best embedding for future FlowNet similarity checks.
        """

        # If tracker doesn't exist yet for this uuid, create one
        if uuid not in self.trackers:
            self.trackers[uuid] = FlowTracker(flow_net=self._get_flow_net(), uuid=uuid)

        tracker = self.trackers[uuid]

        # Only update tracking info if similarity is above threshold
        if similarity >= SIMILARITY_THRESHOLD:
            frame_index = self._get_frame_index_from_frame(frame)

            # Update current state
            tracker.last_box = box
            tracker.last_frame_index = frame_index


            #logging.info( f"[FlowNet] Tracking box updated for UUID {uuid} at frame {frame_index} | similarity: {similarity:.2f}%")

            # Update best score and embedding if this is a better match
            if similarity > tracker.best_score:
                tracker.best_score = similarity
                """
                if embedding is not None:
                    tracker.best_embedding = embedding
                logging.info(f"[FlowNet] Best score updated for UUID {uuid} → {similarity:.2f}%")
                """
            logging.info(f"[FlowNet-IoU] Updated tracker for UUID {uuid} at frame {frame_index} | similarity: {similarity:.2f}%")

    """
    def match_or_add(self, box, similarity, frame, uuid,SIMILARITY_THRESHOLD):
        
        Register or update a person for tracking.
        If uuid not tracked yet, add it. If already tracked, update score.
        
        if uuid not in self.trackers:
            self.trackers[uuid] = FlowTracker(flow_net=self._get_flow_net(), uuid=uuid)
            #self.trackers[uuid].last_box = box
            #self.trackers[uuid].last_frame_index = self._get_frame_index_from_frame(frame)
        tracker = self.trackers[uuid]

        # Always update the current tracking box if similarity is sufficient
        if similarity >= SIMILARITY_THRESHOLD:
            tracker.last_box = box
            tracker.last_frame_index = self._get_frame_index_from_frame(frame)
            logging.info(f"[FlowNet] Tracking box updated for UUID {uuid} | similarity: {similarity:.2f}%")

            # Only update best similarity score if improved
            if similarity > tracker.best_score:
                tracker.best_score = similarity
                logging.info(f"[FlowNet] Best score updated for UUID {uuid} → {similarity:.2f}%")
    """

    def update_all(self, frame_index):
        """
        Update all tracked boxes using FlowNet.
        """
        for tracker in self.trackers.values():
            tracker.update_track_frame(frame_index)

    def get_all(self):
        """
        Return all tracker objects (used for drawing).
        """
        return self.trackers.values()


    def _get_frame_index_from_frame(self, frame):
        # Add custom logic if needed, or attach frame index externally
        return getattr(frame, 'frame_index', None)

    def _get_flow_net(self):
        # If you want to use a shared FlowNet instance, override this logic
        from server.FlowNet_Component.SimpleFlowNet import SimpleFlowNet
        import logging
        return SimpleFlowNet(logger=logging.getLogger(__name__))



""""
class TrackingManager:
    def __init__(self):
        #self.tracked_people = [] ######################### list of all currently tracked people
        self.trackers = {}  # uuid -> FlowTracker

    def update_all(self, current_frame):
        for person in self.tracked_people:
            person.update_frame(current_frame) ########### Update the box of each tracked person using FlowNet

    #################### check match of the new detection to an existing tracked person
    def match_or_add(self, new_box, new_score, frame, person_id=""):
        threshold_iou = 0.3  ############################# Minimum IoU to consider a match
        for person in self.tracked_people:
            if self._box_iou(person.box, new_box) > threshold_iou:
                ########################################## If matched update the tracked box and best score
                person.maybe_update_match(new_score, new_box)
                return

        ################################################## No existing match found — add as a new tracked person
        self.tracked_people.append(TrackedPerson(new_box, new_score, frame, person_id))

    def get_all(self):
        return self.tracked_people ####################### Return all tracked people

    def _box_iou(self, boxA, boxB):
        # Compute IoU between two boxes
        ax, ay, aw, ah = boxA
        bx, by, bw, bh = boxB

        xA = max(ax, bx)
        yA = max(ay, by)
        xB = min(ax + aw, bx + bw)
        yB = min(ay + ah, by + bh)

        inter_area = max(0, xB - xA) * max(0, yB - yA)
        boxA_area = aw * ah
        boxB_area = bw * bh

        iou = inter_area / float(boxA_area + boxB_area - inter_area + 1e-5)
        return iou

"""
# server/FlowNet_Componenet/TrackedPerson.py

from server.FlowNet_Componenet.FlowNet_Utils import FlowTracker

class TrackedPerson:
    def __init__(self, box, score, frame, person_id=""):
        self.flow_tracker = FlowTracker()
        self.box = box  # (x, y, w, h)
        self.best_score = score  # Lower is better
        self.person_id = person_id
        self.frame = frame

    def update_frame(self, new_frame):
        self.box = self.flow_tracker.update_track_frame(new_frame, self.box)
        self.frame = new_frame

    def maybe_update_match(self, new_score, new_box):
        if new_score < self.best_score:
            self.best_score = new_score
            self.box = new_box


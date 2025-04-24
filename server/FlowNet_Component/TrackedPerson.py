from server.FlowNet_Component.FlowNet_Utils import FlowTracker

class TrackedPerson:
    def __init__(self, box, score, frame, person_id=""):
        self.flow_tracker = FlowTracker()    ############# Each tracked person has their own optical flow tracker
        self.box = box      ############################## Current bounding box for this person (x, y, w, h)
        self.best_score = score   ######################## Best similarity score (FaceNet match) observed so far
        self.person_id = person_id  ###################### For this person
        self.frame = frame  ############################## The last frame where this person was seen

    ###################################################### Update the tracked box based on motion between frames
    def update_frame(self, new_frame):
        self.box = self.flow_tracker.update_track_frame(new_frame, self.box)
        self.frame = new_frame
    ###################################################### Update if the new match is better (lower distance)
    def maybe_update_match(self, new_score, new_box):
        if new_score < self.best_score:
            self.best_score = new_score
            self.box = new_box


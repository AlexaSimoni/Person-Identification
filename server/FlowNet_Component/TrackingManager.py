# server/FlowNet_Component/TrackingManager.py

from server.FlowNet_Component.TrackedPerson import TrackedPerson

class TrackingManager:
    def __init__(self):
        self.tracked_people = [] ######################### list of all currently tracked people

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

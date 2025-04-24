# server/FlowNet_Component/TrackingManager.py

from server.FlowNet_Component.TrackedPerson import TrackedPerson

class TrackingManager:
    def __init__(self):
        self.tracked_people = []

    def update_all(self, current_frame):
        for person in self.tracked_people:
            person.update_frame(current_frame)

    def match_or_add(self, new_box, new_score, frame, person_id=""):
        threshold = 50  # Distance threshold to consider a match

        matched = False
        for person in self.tracked_people:
            iou = self._box_iou(person.box, new_box)
            if iou > 0.3:
                person.maybe_update_match(new_score, new_box)
                matched = True
                break

        if not matched:
            new_person = TrackedPerson(new_box, new_score, frame, person_id)
            self.tracked_people.append(new_person)

    def get_all(self):
        return self.tracked_people

    def _box_iou(self, boxA, boxB):
        # Compute Intersection over Union
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

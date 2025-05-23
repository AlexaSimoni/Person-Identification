import base64
import cv2


class Detection:
    """
    A class to represent a single detection.
    """

    def __init__(self, coordinates, confidence, image_patch, frame_index):
        self.founded = False
        self.similarity = None
        self.coordinates = coordinates
        self.confidence = confidence
        self.width = coordinates[2] - coordinates[0]
        self.height = coordinates[3] - coordinates[1]
        self.image_base_64 = self.encode_image_to_base64(image_patch)  # Store the image as base64
        self.frame_index = frame_index  # Store the frame index where the detection occurred

    def __str__(self):
        """
        Return a string representation of the detection.
        """
        return (f"Object type: Person\n"
                f"Coordinates: {self.coordinates}\n"
                f"Width: {self.width}, Height: {self.height}\n"
                f"Probability: {self.confidence}\n"
                f"Image Base64: {self.image_base_64[:30]}...\n")  # Show the first 30 characters of the base64 string

    @staticmethod
    def encode_image_to_base64(image_patch):
        """
        Encode an image patch to base
        """
        if image_patch is not None:
            _, buffer = cv2.imencode('.jpg', image_patch)
            image_patch_bytes = buffer.tobytes()
            return base64.b64encode(image_patch_bytes).decode('utf-8')
        return None

    def to_dict(self):
        """
        Convert the detection to a dictionary
        """
        return {
            "coordinates": self.coordinates,
            "confidence": self.confidence,
            "width": self.width,
            "height": self.height,
            "frame_index": self.frame_index,
            "image_base_64": self.image_base_64
        }

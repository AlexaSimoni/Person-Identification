import os
from os.path import join
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
FACENET_SERVER_PORT = os.getenv("FACENET_SERVER_PORT", 8001)
FACENET_SERVER_URL = os.getenv("FACENET_SERVER_URL", f"http://localhost:{FACENET_SERVER_PORT}")
FACENET_FOLDER = os.getenv("FACENET_FOLDER",join(ROOT_PATH, "FaceNet_Componenet"))
YOLO_SERVER_PORT = os.getenv("YOLO_SERVER_PORT", 8000)
YOLO_SERVER_URL = os.getenv("YOLO_SERVER_URL", f"http://localhost:{YOLO_SERVER_PORT}")
YOLO_FOLDER = os.getenv("YOLO_FOLDER",join(ROOT_PATH, "Yolo_Componenet"))
SIMILARITY_THRESHOLD = os.getenv("SIMILARITY_THRESHOLD", 30.0)
#MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://root:example@localhost:27017/?authMechanism=DEFAULT")
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
FLOWNET_FOLDER = os.getenv("FLOWNET_FOLDER", join(ROOT_PATH, "FlowNet_Component", "FlowNetPytorch"))
#FLOWNET_MODEL_PATH = os.getenv("FLOWNET_MODEL_PATH", join(ROOT_PATH, "FlowNet_Component", "checkpoints", "FlowNetS_checkpoint.pth.tar"))
FLOWNET_MODEL_PATH = os.getenv("FLOWNET_MODEL_PATH", join(ROOT_PATH, "..", "FlowNet_Component", "checkpoints", "flownets_from_caffe.pth"))
#USE_FLOWNETS = os.getenv("USE_FLOWNETS", "false").lower() == "true"

# or False to disable
CLEAR_UUID_HISTORY_BEFORE_RUN = False
# or False to disable
CLEAR_ALL_HISTORY_BEFORE_RUN = True

# Controls whether FlowNet tracking logic should be used at all
ENABLE_FLOWNET_TRACKING = True
# Controls using FLOWNETS - flownets_from_caffe
USE_FLOWNETS = True
#ENABLE_CLIP_POST_FILTERING: bool = True

# Set to False to disable CLIP filtering and reference updating
USE_CLIP_IN_FLOWTRACKING = True
# If True, FlowNet updates from every FaceNet detection above threshold
# If False, FlowNet initializes only once, then runs independently
FLOWNET_MATCH_FROM_FACENET_EVERY_TIME = True
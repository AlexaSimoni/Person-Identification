import logging
import multiprocessing
import os
from typing import Dict, Any
import ffmpeg
import cv2
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from server.FaceNet_Componenet.FaceNet_Utils import embedding_manager, face_embedding
from server.Utils.db import detected_frames_collection, embedding_collection
from server.Yolo_Componenet.YoloV8Detector import YoloV8Detector
from server.config.config import FACENET_SERVER_URL, MONGODB_URL, SIMILARITY_THRESHOLD
from motor.motor_asyncio import AsyncIOMotorClient
import threading
import queue
from server.FlowNet_Component.FlowNet_Utils import start_flow_tracking, get_tracking_manager, draw_tracking_boxes
#from server.Utils.framesGlobals import annotated_frames, detections_frames, all_even_frames, dir_path
import server.Utils.framesGlobals as framesGlobals
from server.FlowNet_Component.FlowNet_Utils import insert_flowdetected_frames
#from server.FlowNet_Component.TrackingManager import tracking_manager


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

detector = YoloV8Detector("../yolov8l.pt", logger)
face_comparison_server_url = FACENET_SERVER_URL + "/compare/"
client = AsyncIOMotorClient(MONGODB_URL)
tracker_manager = get_tracking_manager()

#List to store processed frames and their indices
"""
annotated_frames = {}
detections_frames = {}
all_even_frames = {}
"""
#made as globals

#Initialize a queue for frames to be annotated
frame_queue = queue.Queue()

#Clear old frame states
framesGlobals.all_even_frames.clear()
framesGlobals.detections_frames.clear()
framesGlobals.annotated_frames.clear()
#dir_path.clear()

async def insert_detected_frames_separately(uuid: str, running_id: str, detected_frames: Dict[str, Any],
                                            frame_per_second: int = 30):
    #Insert detected frames separately into the MongoDB collection
    for frame_index, frame_data in detected_frames.items():
        frame_document = {
            "uuid": uuid,
            "running_id": running_id,
            "frame_index": frame_index,
            "frame_data": frame_data,
            "embedded": False,
            "frame_per_second": frame_per_second
        }
        await detected_frames_collection.insert_one(frame_document)

def annotate_frame_worker(similarity_threshold, detected_frames, uuid, reference_embeddings):
    #Worker function to annotate frames with detected faces
    while True:
        try:
            item = frame_queue.get()
            if item is None:
                break

            frame, frame_obj, frame_index = item
            #Annotate the frame
            logger.info(f"Annotating frame {frame_obj.frame_index}")
            annotate_frame(frame, frame_obj, similarity_threshold, detected_frames, uuid,
                           reference_embeddings, frame_index)
            #added for flownet bbox
            #if frame_index % 2 == 0:
            #    draw_tracking_boxes(frame, frame_index)
            draw_tracking_boxes(frame, frame_index)

            # Safely store the annotated frame in the shared dictionary
            framesGlobals.annotated_frames[frame_index] = frame

        except Exception as e:
            logger.error(f"Error in annotate_frame_worker: {e}")

        finally:
            logger.info(f"Finished processing frame {frame_index}, marking as done")
            frame_queue.task_done()

async def process_and_annotate_video(video_path: str, similarity_threshold: float, uuid: str, running_id: str) -> str:
    cap = cv2.VideoCapture(video_path) ### open video
    frame_per_second = int(cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Error opening video file")
    print_to_log_video_parameters(cap)

    #reference_embeddings = await embedding_manager.get_reference_embeddings(uuid)
    reference_embeddings = {"data": await embedding_manager.get_reference_embeddings(uuid)}

    dir_temp=os.path.dirname(video_path)
    if not dir_temp or not os.path.isdir(dir_temp):
        raise HTTPException(status_code=500, detail="Invalid or missing directory path (dir_temp)")
    logger.info(f"[yolo_utils] temp dir set to: {dir_temp}")
    framesGlobals.dir_path = os.path.join(dir_temp,"flow_debug")
    os.makedirs(framesGlobals.dir_path, exist_ok=True)
    if not framesGlobals.dir_path or not os.path.isdir(framesGlobals.dir_path):
        raise HTTPException(status_code=500, detail="Invalid or missing directory path (dir_path)")
    logger.info(f"[yolo_utils_flow] Global dir_path set to: {framesGlobals.dir_path}")



    output_path = video_path.replace(".mp4", "_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using mp4v codec for MPEG-4
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    if not out.isOpened():
        cap.release()
        raise HTTPException(status_code=500, detail="Error initializing video writer")

    frame_index = 0
    detected_frames: Dict[str, Any] = {}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Start a few annotation worker threads
    num_annotation_threads = multiprocessing.cpu_count()
    threads = []
    for i in range(num_annotation_threads):
        t = threading.Thread(target=annotate_frame_worker,
                             args=(similarity_threshold, detected_frames, uuid, reference_embeddings))
        t.start()
        threads.append(t)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % 50 == 0:
            logger.info(f"[yolo_utils] Refreshing reference embeddings at frame {frame_index}")
            #reference_embeddings["data"] = await embedding_manager.get_reference_embeddings(uuid)
            #new_data = await embedding_manager.get_reference_embeddings(uuid)
            #logger.info(f"[yolo_utils] Refreshed embeddings count: {len(new_data.get('embeddings', []))}")
            #reference_embeddings["data"] = new_data

            # Process and save new FaceNet embeddings from detected frames
            new_embeddings = await embedding_manager.process_detected_frames(uuid, face_embedding)
            logger.info(f"[yolo_utils] New FaceNet embeddings added at frame {frame_index}: {len(new_embeddings)}")

            # Reload latest reference embeddings from DB
            new_data = await embedding_manager.get_reference_embeddings(uuid)
            reference_embeddings["data"] = new_data
            logger.info(f"[yolo_utils] Total reference embeddings now: {len(new_data.get('embeddings', []))}")

        frame_index += 1
        framesGlobals.all_even_frames[frame_index] = frame

        #Queue even frames for annotation processing
        if frame_index % 2 == 0:
            #Directly add even frames for flownet use
            #framesGlobals.all_even_frames[frame_index] = frame
            #Detect faces
            frame_obj = detector.predict(frame, frame_index=frame_index)
            #Queue the frame for annotation
            frame_queue.put((frame, frame_obj, frame_index))

            logger.info(f"Processing frame {frame_index}/{total_frames}")
        else:
            #Directly add odd frames to the annotated_frames dictionary
            framesGlobals.annotated_frames[frame_index] = frame

    #Wait for all frames to be processed
    frame_queue.join()
    logger.info("All frames processed")

    #Stop the worker threads
    for _ in range(num_annotation_threads):
        frame_queue.put(None)
    for t in threads:
        t.join()

    #Write the frames to output video
    for index in range(total_frames):
        frame = framesGlobals.annotated_frames.get(index)
        if frame is not None:
            #if index % 2 == 1:
                #check_and_annotate(index, frame)
            out.write(frame)

    cap.release()
    out.release()
    logger.info(f"Video processing complete , output file saved at {output_path}")
    #Save detected frames to MongoDB separately
    await insert_detected_frames_separately(uuid=uuid, running_id=running_id, detected_frames=detected_frames,
                                            frame_per_second=frame_per_second)
    await insert_flowdetected_frames(uuid=uuid, running_id=running_id, frame_per_second=frame_per_second)

    #Re-encode the annotated video
    reencoded_output_path = video_path.replace(".mp4", "_annotated_reencoded.mp4")
    reencode_video(output_path, reencoded_output_path)

    if not os.path.exists(reencoded_output_path):
        raise HTTPException(status_code=500, detail="Re-encoded video file not found after processing")

    return reencoded_output_path

def check_and_annotate(frame_index, frame):
    #Check if the detections in the previous and next frames are similar and annotate the current frame
    #used for gap closeup in frames bounding
    diff_margin = 100
    #check the before and after frame detections and if their coordinates are similar add fiction annotation
    if frame_index - 1 in framesGlobals.detections_frames and frame_index + 1 in framesGlobals.detections_frames:
        #get the coordinates of the detections
        detection1 = framesGlobals.detections_frames[frame_index - 1]
        detection2 = framesGlobals.detections_frames[frame_index + 1]
        #check if the coordinates are similar by a margin of error
        if abs(detection1[0][0] - detection2[0][0]) < diff_margin and abs(
                detection1[0][1] - detection2[0][1]) < diff_margin:
            #add the coordinates to the frame
            x1, y1, x2, y2 = detection1[0]
            #Ensure coordinates are within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            #Draw bounding box in red
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            #Position text above the bounding box and ensure it fits within the frame
            text = f"{detection1[1]:.2f}%"
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 0.8
            font_thickness = 2

            #Calculate text size and position
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10

            #Draw background rectangle for text
            cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5),
                          (text_x + text_size[0], text_y + 5), (0, 0, 255), cv2.FILLED)

            #Draw text in white
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
            return True

def wrapper(data):
    #Wrapper function to calculate similarity between embeddings
    return embedding_manager.calculate_similarity(
        data[0],
        data[1]
    )

def annotate_frame(frame, frame_obj, similarity_threshold, detected_frames, uuid, reference_embeddings, frame_index):
    #Annotate a frame with detected faces, compute similarity, and register FlowNet tracking
    logger.info(f"Found in frame {frame_obj.frame_index}: {len(frame_obj.detections)} detections")
    #Pair each detection with the reference embeddings
    #datas = [(reference_embeddings, detection.image_base_64) for detection in frame_obj.detections]
    datas = [(reference_embeddings.get("data"), detection.image_base_64) for detection in frame_obj.detections]

    similarities = [wrapper(data) for data in datas]

    for detection, similarity in zip(frame_obj.detections, similarities):
        if similarity is not None and similarity > similarity_threshold:
            #tracking_manager.try_save_initial_clip_reference(uuid, frame_index, box)
            x1, y1, x2, y2 = detection.coordinates
            framesGlobals.detections_frames[frame_index] = (detection.coordinates, similarity)

            #Ensure coordinates are within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            #Draw bounding box in red
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            #Draw similarity score label
            text = f"{similarity:.2f}%"
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 0.8
            font_thickness = 2

            #Calculate text size and position
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10

            #Draw background rectangle for text
            cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5),
                          (text_x + text_size[0], text_y + 5), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

            #Register FlowNet tracker
            box = (x1, y1, x2 - x1, y2 - y1)  # convert to (x, y, w, h)

           # tracking_manager = get_tracking_manager()

            #tracker_manager.match_or_add(box, similarity, frame, uuid,SIMILARITY_THRESHOLD=similarity_threshold)
            tracker_manager.match_or_add(box, similarity, frame_index, uuid, SIMILARITY_THRESHOLD=similarity_threshold)
            # Save initial CLIP reference from FaceNet detection (only once)

            tracker_manager.try_save_initial_clip_reference(uuid, frame_index, box)
            logger.info(f"Similarity score: {similarity:.2f}% for detection: {detection.frame_index}, Accepted")

            #Ensure FlowNet thread is running
            start_flow_tracking()

            detection.similarity = similarity
            detection.founded = True
            detected_frames[f"frame_{frame_obj.frame_index}"] = {"cropped_image": detection.image_base_64,
                                                                 "similarity": similarity}
            break
        else:
            logger.debug(f"No similarity score or below threshold for detection: {detection.frame_index}")


async def calculate_similarity(uuid, detected_image_base64):
    #Calculate similarity between detected image and reference embeddings for the given UUID
    reference_embeddings = await embedding_manager.get_reference_embeddings(uuid)
    similarity = await embedding_manager.calculate_similarity(
        reference_embeddings,
        detected_image_base64,
        face_embedding
    )
    return similarity

def create_streaming_response(file_path: str, filename: str):
    #Return a StreamingResponse to send the annotated video file as a downloadable attachment
    logger.info(f"Creating streaming response for file: {file_path}")
    return StreamingResponse(
        iter_file(file_path),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

#Async generator to yield video file chunks
async def iter_file(file_path: str):
    #Async generator to yield video file chunks
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, mode="rb") as file_like:
        while chunk := file_like.read(1024):
            yield chunk


async def fetch_detected_frames(uuid: str, running_id: str):
    #Retrieve all previously detected frames from MongoDB for the given UUID and running_id
    #Used by the get_detected_frames API
    cursor = detected_frames_collection.find({"uuid": uuid, "running_id": running_id})
    frame_per_second = 30   #default
    detected_frames = {}
    async for document in cursor:
        frame_index = document["frame_index"]
        frame_data = document["frame_data"]

        detected_frames[frame_index] = frame_data

    #Append user metadata if available
    if detected_frames:
        frame_per_second = document["frame_per_second"]
    extra_details = await embedding_collection.find_one({"uuid": uuid})
    if extra_details:
        detected_frames["user_details"] = extra_details["user_details"]
        detected_frames["frame_per_second"] = frame_per_second

    return detected_frames

# ffmpeg re-encoding of video
def reencode_video(input_path, output_path):
    #Re-encode the annotated video using ffmpeg to ensure compatibility and compression
    try:
        logger.info(f"Checking if {input_path} exists...")
        if os.path.exists(input_path):
            logger.info(f"{input_path} exists.")
        else:
            logger.error(f"{input_path} does NOT exist.")

        logger.info(f"Checking if {output_path} is accessible...")
        if os.access(output_path, os.R_OK):
            logger.info(f"{output_path} is readable.")
        else:
            logger.error(f"{output_path} is NOT readable.")

        # Ensure the input file exists
        if not os.path.exists(input_path):
            logger.error(f"Input file does not exist: {input_path}")
            return

        logger.info(f"Input file confirmed: {input_path}")
        logger.info("Re-encoding video...")

        # Run ffmpeg command
        process = (
            ffmpeg
            .input(input_path)
            .output(output_path, vcodec='libx264', acodec='aac', strict='-2')
            .run(capture_stdout=True, capture_stderr=True)
        )
        logger.info("Video re-encoded successfully!")

    except Exception as e:
        logger.error(f"ffmpeg error: {e.stderr.decode('utf-8')}")
        logger.error(f"Error occurred during re-encoding: {e}")
        logger.error(f"An unexpected error occurred during re-encoding: {e}")


def print_to_log_video_parameters(cap):
    #Print video metadata (frame count, resolution, FPS)
    logger.info(f"Number of frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    logger.info(f"Frame width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    logger.info(f"Frame height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    logger.info(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")

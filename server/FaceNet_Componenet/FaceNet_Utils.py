import base64
import logging
from typing import List, Dict, Union
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import io
from fastapi import UploadFile
from numpy import floating
from numpy._typing import _16Bit
from sklearn.metrics.pairwise import cosine_similarity

from server.Utils.db import embedding_collection
import torch
from server.Utils.db import detected_frames_collection
from server.config.config import SIMILARITY_THRESHOLD

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceEmbedding:
    #FaceEmbedding class to handle face detection and embedding calculation
    #The class uses MTCNN for face detection and InceptionResnetV1 for embedding calculation
    #Can be used to calculate embeddings for a single image or compare embeddings of two faces
    def __init__(self, device: str):
        self.device = device
        self.mtcnn = MTCNN(device=device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        logger.info("MTCNN and InceptionResnetV1 models loaded successfully\n FaceEmbedding instance created")

    @staticmethod
    def preprocess_image(image_base64: str) -> Image.Image:
        #Preprocess the image by decoding the base64 string
        #image_base64: Base64 encoded image
        #return: Decoded image
        decoded_image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
        return decoded_image

    def get_embedding(self, image: Image.Image) -> Union[np.ndarray, None]:
        #Get the embedding of the face in the image
        #image: Image to process
        #return: Embedding of the face in the image if a face is detected, otherwise None
        try:
            img_cropped = self.mtcnn(image)
            if img_cropped is not None:
                img_cropped = img_cropped.to(self.device)  # Ensure the tensor is on the same device as the model
                # Calculate embedding (unsqueeze to add batch dimension)
                img_embedding = self.resnet(img_cropped.unsqueeze(0)).detach().cpu().numpy()
                logger.debug("Face detected and embedding calculated successfully")
                return img_embedding
            logger.warning("No faces detected in the image.")
        except Exception as e:
            logger.error(f"Error in get_embedding: {e}")
        return None

    @staticmethod
    def compare_faces_embedding(embedding: np.ndarray, embedding_list: List[np.ndarray]) -> float:
        #Compare the input embedding with a list of embeddings and return the maximum similarity percentage
        #embedding: Face embedding to compare
        #embedding_list: List of embeddings to compare against
        #return: Maximum similarity percentage

        # Ensure that both embeddings are 2D arrays
        embedding = embedding.reshape(1, -1)
        embedding_array = np.array(embedding_list).reshape(len(embedding_list), -1)

        # Compute cosine similarities in a vectorized manner
        similarities = cosine_similarity(embedding, embedding_array)

        # Convert to percentages
        similarity_percentages = similarities[0] * 100
        similarity_percentages: floating[_16Bit] = np.round(similarity_percentages, 2)

        # Log the similarity percentages (optional)
        logger.info(f"Similarity percentages:\n {similarity_percentages}")

        # Return the maximum similarity percentage
        return np.max(similarity_percentages)


class EmbeddingManager:
    #EmbeddingManager class to manage reference embeddings and user details in the database
    #Can be used to save embeddings to the database, process detected frames, and calculate similarity
    def __init__(self, collection):
        self.collection = collection

    async def save_embeddings_to_db(self, uuid: str, new_embeddings: List[np.ndarray], user_details={}):
        #Save the new embeddings to the database
        existing_record = await self.collection.find_one({"uuid": uuid})
        if existing_record:
            update_data = await self.handle_existing_record(existing_record, new_embeddings, user_details)
        else:
            update_data = self.handle_new_record(new_embeddings, user_details)

        await self.collection.update_one(
            {"uuid": uuid},
            {"$set": update_data},
            upsert=True
        )
        logger.info("Reference embeddings and average embedding calculated successfully")

    async def process_detected_frames(self, uuid: str, face_embedding: FaceEmbedding) -> List[np.ndarray]:
        #Process detected frames and calculate embeddings for frames with similarity
        #greater than 80% and not embedded
        cursor = detected_frames_collection.find(
        {"uuid": uuid, "embedded": False, "frame_data.similarity": {"$gt": 80}})    #similarity rate
        #{"uuid": uuid, "embedded": False, "frame_data.similarity": {"$gt": SIMILARITY_THRESHOLD}})    #similarity rate

        new_embeddings = []
        existing_embeddings = await self.get_existing_embeddings(uuid)
        async for doc in cursor:
            await detected_frames_collection.update_one({"_id": doc["_id"]}, {"$set": {"embedded": True}})
            cropped_image_base64 = doc["frame_data"]["cropped_image"]
            cropped_image = face_embedding.preprocess_image(cropped_image_base64)
            embedding = face_embedding.get_embedding(cropped_image)
            if embedding is not None:
                new_embeddings.append(embedding)
        return new_embeddings

    async def get_existing_embeddings(self, uuid: str) -> List[np.ndarray]:
        #Get the existing embeddings for a given UUID
        record = await self.collection.find_one({"uuid": uuid})
        if record:
            return [np.array(e) for e in record.get("embeddings", [])]
        return []

    def is_unique_embedding(self, new_embedding: np.ndarray, existing_embeddings: List[np.ndarray],
                            threshold: float = 0.2) -> bool:
        #Check if the new embedding is unique compared to the existing embeddings
        for existing_embedding in existing_embeddings:
            similarity = self.compare_embeddings(new_embedding, existing_embedding)
            if similarity >= 70.0:
                return False
        return True

    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        #Compare two embeddings and return the similarity percentage
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
        similarity = cosine_similarity(emb1, emb2)[0][0] * 100
        return similarity

    async def process_images(self, files: List[UploadFile], face_embedding: FaceEmbedding) -> List[np.ndarray]:
        #Process the uploaded images and calculate embeddings
        #Can be used to process reference images
        embeddings = []
        for file in files:
            image_bytes = await file.read()
            reference_image = Image.open(io.BytesIO(image_bytes))
            logger.debug("Reference image loaded successfully")
            embedding = face_embedding.get_embedding(reference_image)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                logger.error(f"Failed to calculate embedding for image: {file.filename}")
        return embeddings

    async def handle_existing_record(self, existing_record: Dict, new_embeddings: List[np.ndarray],
                                     user_details: Dict) -> Dict:
        #Handle the case when a record already exists in the database
        existing_user_details = self.merge_user_details(existing_record, user_details)
        existing_embeddings = self.convert_to_numpy(existing_record.get("embeddings", []))
        unique_embeddings = self.filter_unique_embeddings_dynamically(new_embeddings, existing_embeddings)
        embeddings_to_save = existing_embeddings + unique_embeddings
        logger.info(f"Unique embeddings: {len(embeddings_to_save)}")
        average_embedding = self.calculate_average_embedding(embeddings_to_save, existing_record)

        return {
            "embeddings": [e.tolist() for e in embeddings_to_save],
            "average_embedding": average_embedding,
            "user_details": existing_user_details
        }

    def handle_new_record(self, new_embeddings: List[np.ndarray], user_details: Dict) -> Dict:
        #Handle the case when a record does not exist in the database
        average_embedding = self.calculate_average_embedding(new_embeddings)

        return {
            "embeddings": [e.tolist() for e in new_embeddings],
            "average_embedding": average_embedding,
            "user_details": user_details
        }

    @staticmethod
    def merge_user_details(existing_record: Dict, new_user_details: Dict) -> Dict:
        #Merge the new user details with the existing user details in the record
        existing_user_details = existing_record.get("user_details", {})
        existing_user_details.update(new_user_details)
        return existing_user_details

    @staticmethod
    def convert_to_numpy(embeddings: List[Union[List, np.ndarray]]) -> List[np.ndarray]:
        #Convert a list of embeddings to numpy arrays
        return [np.array(e) for e in embeddings]

    @staticmethod
    def filter_unique_embeddings_dynamically(new_embeddings: List[np.ndarray], existing_embeddings: List[np.ndarray],
                                             threshold: float = 0.1) -> List[np.ndarray]:
        #Filter unique embeddings from a list of new embeddings based on a dynamic threshold
        unique_embeddings = []

        for new_emb in new_embeddings:
            is_unique = True
            for existing_emb in existing_embeddings:
                similarity = cosine_similarity(new_emb.reshape(1, -1), existing_emb.reshape(1, -1))[0][0]
                if similarity >= (1 - threshold):
                    is_unique = False
                    break
            if is_unique:
                existing_embeddings.append(new_emb)
                unique_embeddings.append(new_emb)

        return unique_embeddings

    @staticmethod
    def calculate_average_embedding(embeddings: List[np.ndarray], existing_record: Dict = None) -> List:
        #Calculate the average embedding from a list of embeddings
        if embeddings:
            return np.mean(embeddings, axis=0).tolist()
        elif existing_record:
            return existing_record.get("average_embedding", [])
        return []

    async def get_reference_embeddings(self, uuid: str):
        #Get the reference embeddings for a given UUID From the database
        return await self.collection.find_one({"uuid": uuid})

    def calculate_similarity(self, record, detected_image_base64: str) -> float:
        #Calculate the similarity between the detected image and the reference embeddings
        embeddings = [np.array(e) for e in record["embeddings"]]
        detected_image = face_embedding.preprocess_image(detected_image_base64)
        detected_embedding = face_embedding.get_embedding(detected_image)
        if detected_embedding is not None:
            return face_embedding.compare_faces_embedding(embedding=detected_embedding, embedding_list=embeddings)
        else:
            logger.debug("Face not detected in the image")
            return 0

###unused?
def initialize_model(device):
    #Initialize the MTCNN and InceptionResnetV1 models
    global mtcnn, model
    mtcnn = MTCNN(device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print("Model and MTCNN initialized in process.")

#Initialize FaceEmbedding and EmbeddingManager instances
try:
    face_embedding = FaceEmbedding(device="cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    logger.error(f"Error initializing FaceEmbedding: {e}")
    face_embedding = None  # Handle fallback logic if needed

embedding_manager = EmbeddingManager(embedding_collection)

from motor.motor_asyncio import AsyncIOMotorClient
from server.config.config import MONGODB_URL
from server.Utils.framesGlobals import flow_clip_reference
import logging
async_client = AsyncIOMotorClient(MONGODB_URL)
async_database = async_client["Person_Recognition"]
embedding_collection = async_database.get_collection("embeddings")
detected_frames_collection = async_database.get_collection("detected_frames")
logger = logging.getLogger(__name__)


async def check_mongo():
    """
    Health check function to verify that the MongoDB connection is up and running.
    """
    try:
        # The ping command is used to check if the connection to MongoDB is up and running
        async_client.admin.command('ping')
        return True
    except ConnectionError:
        return False


#def delete_many_embedding_collection(query={}):
    """
    Delete many documents from the embedding collection.
    """
    #embedding_collection.delete_many(query)

async def delete_many_embedding_collection(query={}):
    """
    Asynchronously delete many documents from the embedding collection.
    """
    result = await embedding_collection.delete_many(query)
    return result.deleted_count

#def delete_many_detected_frames_collection(query={}):
    """
    Delete many documents from the detected frames collection.
    """
#   detected_frames_collection.delete_many(query)

async def delete_many_detected_frames_collection(query={}):
    """
    Asynchronously delete many documents from the detected frames collection.
    """
    result = await detected_frames_collection.delete_many(query)
    return result.deleted_count

async def clear_all_user_data(uuid: str):
    logger.info(f"[DB Manager] Clearing MongoDB + memory data for UUID: {uuid}")

    # Delete frame data
    deleted_frames = await delete_many_detected_frames_collection({"uuid": uuid})
    logger.info(f"[DB Manager] Deleted {deleted_frames} frames from DB")

    # Clear embeddings
    update_result = await embedding_collection.update_one(
        {"uuid": uuid},
        {"$set": {"embeddings": []}}
    )
    if update_result.modified_count:
        logger.info(f"[DB Manager] Cleared embeddings for UUID {uuid}")

    # Clear CLIP memory cache
    if uuid in flow_clip_reference:
        del flow_clip_reference[uuid]
        logger.info(f"[DB Manager] Removed in-memory CLIP ref for UUID {uuid}")

async def clear_all_data():
    logger.info("[DB Manager] Clearing data for ALL UUIDs")

    # Remove all detected frames
    frames_deleted = await delete_many_detected_frames_collection({})
    logger.info(f"[DB Manager] Deleted {frames_deleted} detected-frames docs")

    # Remove all embeddings
    embeddings_deleted = await delete_many_embedding_collection({})
    logger.info(f"[DB Manager] Deleted {embeddings_deleted} embeddings docs")

import os
import certifi
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("uvicorn.error")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "ai_dataset_diagnosis_db"


class Database:
    client: AsyncIOMotorClient = None


db = Database()


async def connect_to_mongo():
    try:
        # Apply TLS certificate only when connecting to MongoDB Atlas (SRV URIs)
        if "mongodb+srv://" in MONGO_URI or "atlas" in MONGO_URI.lower():
            db.client = AsyncIOMotorClient(
                MONGO_URI,
                serverSelectionTimeoutMS=2000, # Short timeout for faster startup
                tlsCAFile=certifi.where(),
            )
        else:
            db.client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        
        # We NO LONGER ping here to avoid blocking startup if DB is down.
        # Connections will be established lazily or fail gracefully during requests.
        logger.info("MongoDB client initialized (connectivity will be checked lazily).")
        print("MongoDB client initialized.")
    except Exception as e:
        db.client = None
        logger.error(f"Failed to initialize MongoDB client: {e}")
        print(f"FAILED to initialize MongoDB client: {e}")


async def close_mongo_connection():
    if db.client:
        db.client.close()
        print("Closed MongoDB connection")


def get_database():
    """ Returns the database instance or None if not connected. """
    if db.client is None:
        return None
    return db.client[DB_NAME]

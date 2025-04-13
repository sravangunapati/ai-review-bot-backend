from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = os.getenv('DB_NAME', 'ai_review_bot')

def get_mongo_client():
    """Create and return a MongoDB client instance"""
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        return client
    except Exception as e:
        raise

def get_db():
    """Get the database instance"""
    client = get_mongo_client()
    return client[DB_NAME] 

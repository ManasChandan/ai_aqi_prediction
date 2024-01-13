import StaticFiles.InformationSetup as info
from pymongo.mongo_client import MongoClient
import urllib.parse as parser
import os
# Create a new client and connect to the server


def CreateMongoCollection():
    client = MongoClient(
        info.urls["mongo_db_cluster_collection"] % (
            parser.quote_plus(os.environ['MONGO_USER_NAME']), parser.quote_plus(
                os.environ['MONGO_PASS_WORD'])))
    db = client['ai_model']
    return db['model_info']

global collection
collection = CreateMongoCollection()

def add_model_info(collection, record):
    collection.insert_one(record)
    return "Model Info Added"


def delete_model_info(collection, id):
    collection.delete_one({"_id": id})
    return "Model deleted"


if __name__ == "__main__":
    add_model_info(collection, {"_id": 0, "model": "GradientBoostRegressor", "params": "(tree_height=72)"})
    delete_model_info(collection=collection, id=1)

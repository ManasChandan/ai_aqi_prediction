import CoreUtility.InformationSetup as info
from pymongo.mongo_client import MongoClient
import urllib.parse as parser
import os


class MongoDBUtility:

    def __init__(self):
        self.collection = None
        client = MongoClient(
            info.urls["mongo_db_cluster_collection"] % (
                parser.quote_plus(os.environ['MONGO_USER_NAME']), parser.quote_plus(
                    os.environ['MONGO_PASS_WORD'])))
        db = client['ai_model']
        self.collection = db['model_info']

    def add_model_info(self, record):
        self.collection.insert_one(record)
        return "Model Info Added"

    def delete_model_info(self, id):
        if self.get_model_info(id) is not None:
            self.collection.delete_one({"_id": id})
            return "Model deleted"
        else:
            return "Model Not Found"

    def get_model_info(self, id):
        model = self.collection.find_one({"_id": id})
        return model

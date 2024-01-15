import os
import pandas as pd
import gc
from datetime import date, timedelta
import CoreUtility.Model as model
from CoreUtility.CRUDMongo import MongoDBUtility
import CoreUtility.WeatherDataFetch as weather

global global_model_parameter, mongo
global_model_parameter = []

mongo = MongoDBUtility()

def model_status():
    global global_model_parameter
    print("Model -- " + str(len(global_model_parameter)))

def get_model():
    global global_model_parameter
    print(len(global_model_parameter))
    if len(global_model_parameter) == 0:
        get_model_from_db()
    return global_model_parameter[0]


def get_model_from_db():
    global global_model_parameter, mongo

    print("Get Model From DB")
    last_day = date.today() - timedelta(days=1)
    model_parameter = mongo.get_model_info(id=last_day.strftime("%Y-%m-%d"))
    if model_parameter is None:
        update_model()
    else:
        print("Fetching Weather Data")
        input_df = weather.pipeline_function()

        global_model_parameter.insert(0, model.GenerateModelPipeline(
            model.CreateModel(model_parameter["model_name"], model_parameter["params"]), input_df.copy()))

        del input_df
        gc.collect()

    print("Model Collected")


def update_model():
    global global_model_parameter, mongo

    print("Fetching Weather Data")
    input_df = weather.pipeline_function()

    print("Calculation Best Model")
    best_model = model.pipeline_function(input_df=input_df.copy())

    print(f"Preparing Best Model - {best_model}")
    global_model_parameter.insert(0, model.GenerateModelPipeline(
        model.CreateModel(best_model[0], best_model[1]), input_df.copy()))

    today = date.today()
    mongo.add_model_info({"_id": today.strftime("%Y-%m-%d"),
                          "model_name": best_model[0],
                          "params": best_model[1]})

    del input_df
    gc.collect()

    print("Model Updated")


def delete_model_from_db():
    global mongo_obj

    old_date = date.today() - timedelta(days=3)
    mongo.delete_model_info({"_id": old_date.strftime("%Y-%m-%d")})


def predict(data_dict):
    print(data_dict)
    prediction = model.Predict(get_model(), data_dict=data_dict)
    return prediction

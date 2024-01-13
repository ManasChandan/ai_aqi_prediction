import pandas as pd

global global_model_parameter
global_model_parameter = [1]


def get_model():
    return global_model_parameter[0]


def update_model():
    global global_model_parameter
    global_model_parameter[0] += 1
    print("Model Updated")
    return "Model Updated"

import CoreUtility.InformationSetup as info
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import ast
import gc
from sklearn.model_selection import ParameterGrid



def ProcessData(input_df):
    """
    Process data by extracting month and ISO week from date columns.

    Parameters:
    - input_df (pandas DataFrame): The input DataFrame containing date columns.

    Returns:
    - processed_df (pandas DataFrame): The processed DataFrame with month and ISO week columns.
    """

    # Make sure your DataFrame has a 'date' column, adjust accordingly
    if 'date' not in input_df.columns:
        raise ValueError("Input DataFrame must have a 'date' column.")

    # Convert the 'date' column to datetime format
    input_df['date'] = pd.to_datetime(input_df['date'])

    # Extract month and ISO week columns
    input_df['month'] = input_df['date'].dt.month
    input_df['iso_week'] = input_df['date'].dt.isocalendar().week

    # Drop the original date column
    processed_df = input_df.drop(columns=['date'])

    return processed_df


def CreateTrainTestData(df, label_column="aqi", test_size=0.35, random_state=None):
    """
    Create training and testing datasets from a DataFrame.

    Parameters:
    - df (pandas DataFrame): The DataFrame containing features and labels.
    - label_column (str): The name of the column representing the labels.
    - test_size (float): Proportion of the dataset to include in the test split (default is 0.25).
    - random_state (int or None): Seed for random number generation (default is None).

    Returns:
    - X_train, X_test, y_train, y_test: Training and testing sets for features and labels.
    """

    features = df.drop(columns=[label_column])
    labels = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def GetAccuracyDetails(predicted, actual):
    '''
    return: Details for 1 - [(actual-predicted)/(actual)]*100 
    '''

    # Calculate the differences between the two arrays
    differences = np.subtract(np.array(actual), np.array(predicted))

    # Calculate accuracy by dividing differences by array1
    accuracy = 1 - np.divide(abs(differences), np.array(actual))

    # Calculate percentiles of the accuracy
    percentile_25 = np.percentile(accuracy, 25)
    median = np.percentile(accuracy, 50)
    percentile_75 = np.percentile(accuracy, 75)

    return {"25": percentile_25, "50": median, "75": percentile_75}


def CreateModel(model_name, params):
    """
    Returns:
    - model: An instance of the specified model with the provided parameters.
    """

    print(model_name, params)

    if type(params) == str:
        print("category_changed")
        params = ast.literal_eval(params)
    
    print(type(params))

    if model_name == 'LinearRegression':
        model = LinearRegression(**params)
    elif model_name == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor(**params)
    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model

def TrainModel(model, X_train, y_train):
    '''
    Return: Trained Model
    '''
    model.fit(X_train, y_train)

    del X_train
    del y_train
    gc.collect()

    return model

def GenerateModelMetrics(model_name, model_grid, X_train, X_test, y_train, y_test):
    '''
    return: Model Metrics
    '''
    model_metric = dict()

    # Iterate through all parameter combinations
    for param_dict in ParameterGrid(model_grid):

        # Create a new model instance with specified parameters
        model = CreateModel(model_name, param_dict)

        # Train the model on the training data
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Store metrics in the dictionary
        model_metric[f"{model_name}|{str(param_dict)}"] = GetAccuracyDetails(
            y_pred, y_test)

    return pd.DataFrame(model_metric).T.reset_index().rename(columns={"index": "model"})


def GetBestModel(eval="50", *model_metrics):
    '''
    return: Best Model
    '''

    # Join all model metrics
    accuracy_df = pd.concat(model_metrics).reset_index(drop=True)

    # get best 50%ile acc
    accuracy_df = accuracy_df.sort_values(
        by=eval, ascending=False).reset_index(drop=True)

    return accuracy_df["model"].iloc[0].split("|")


def Predict(model, data_dict):
    """
    Make a prediction using the provided model and input data.

    Args:
        model: The trained model for making predictions.
        data_dict (dict): A dictionary containing input data.

    Returns:
        dict: A dictionary containing the predicted Air Quality Index (AQI).
    """
    # Define the order of columns for the features
    order_columns = "co,no,no2,o3,so2,pm2_5,pm10,nh3".split(",")

    # Extract features based on the defined order
    features = [data_dict[key] for key in order_columns]

    # Convert the 'date' field to a datetime object
    date = pd.to_datetime(data_dict['date'])

    # Add month and ISO calendar week as additional features
    features.extend([date.month, date.isocalendar()[1]])

    # Make prediction using the model
    prediction = model.predict([features])[0]

    print(prediction)

    # Return the prediction as a dictionary with the 'aqi' key
    return {"aqi": prediction}


def pipeline_function(input_df):
    """
    Execute a data processing and modeling pipeline to find the best model.

    Args:
        input_df (pd.DataFrame): The input DataFrame containing the raw data.

    Returns:
        object: The best model selected based on the pipeline.
    """
    train_test_data = CreateTrainTestData(ProcessData(input_df))

    model_metrices = [GenerateModelMetrics(
        model, model_param, *train_test_data) for model, model_param in info.param_grid.items()]

    best_model = GetBestModel("50", *model_metrices)

    del input_df
    gc.collect()

    return best_model

def GenerateModelPipeline(model, input_df, label_column="aqi"):

    '''
    return: Trained Model
    '''

    processed_df = ProcessData(input_df)

    features = processed_df.drop(columns=[label_column])
    labels = processed_df[label_column]

    return TrainModel(model, features, labels)

if __name__ == "__main__":
    print(ast.literal_eval( "{'max_depth': None}"))
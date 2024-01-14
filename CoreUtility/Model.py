from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import ParameterGrid
import StaticFiles.InformationSetup as info


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

    if type(params) == "str":
        params = ast.literal_eval(params)

    if model_name == 'LinearRegression':
        model = LinearRegression(**params)
    elif model_name == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor(**params)
    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

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
        model_metric[f"{model_name}:{str(param_dict)}"] = GetAccuracyDetails(
            y_pred, y_test)

    return pd.DataFrame(model_metric).T.reset_index().rename(columns={"index": "model"})


def GetBestModel(eval="50", *model_metrics):
    '''
    return: Best Model
    '''

    # Join all model metrics
    accuracy_df = pd.concat(model_metrics).reset_index(drop=True)

    # get best 50%ile acc
    accuracy_df = accuracy_df.sort_values(by=eval, ascending=False).reset_index(drop=True)

    return accuracy_df["model"].iloc[0].split(":")


def pipeline(input_df):
    train_test_data = CreateTrainTestData(ProcessData(input_df))
    model_metrices = [GenerateModelMetrics(
        model, model_param, *train_test_data) for model, model_param in info.param_grid.items()]
    best_model = GetBestModel("50", *model_metrices)
    return best_model

if __name__ == "__main__":
    input_df = pd.read_csv(r"C:\Data Science and DS\DataSets\WeatherData.csv")
    best = pipeline(input_df=input_df)
    print(best)

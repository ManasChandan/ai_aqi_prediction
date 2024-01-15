import os
import requests
import datetime
import pandas as pd
import CoreUtility.InformationSetup as info


def GenerateHistoricalDates():
    '''
    return: current_date - 181, current_date - 1 in unix utc
    '''
    fetch_end_date = datetime.datetime.now()-datetime.timedelta(days=1)
    fetch_start_date = fetch_end_date - datetime.timedelta(days=181)
    return int(fetch_start_date.astimezone(datetime.timezone.utc).timestamp()), int(fetch_end_date.astimezone(datetime.timezone.utc).timestamp())


def FetchDataBetweenDates(start_date, end_date):
    '''
    return: Json of the weather (AQI and Othes) data between dates for Hyderabad
    '''
    params = info.weather_params
    params['start'], params['end'] = start_date, end_date
    params['appid'] = os.environ.get("WEATHER_API_KEY", None)
    session = requests.session()
    response = session.get(info.urls['weather_data'], params=params)
    if response.status_code == 200:
        weather_data = response.json()
    else:
        print(f"Request failed with status code {response.status_code}")
        weather_data = None
    session.close()
    return weather_data


def GenerateWeatherDataFromJson(historical_data_reference):
    '''
    return: dataframe of the json data
    '''
    ref = pd.DataFrame(historical_data_reference['list'])
    componenets = ref['components'].apply(pd.Series)
    aqi = ref['main'].apply(pd.Series)
    date = ref['dt'].apply(pd.Series)
    weather_data = pd.concat([date, componenets, aqi],
                             axis=1).rename(columns={0: "date"})
    weather_data['date'] = pd.to_datetime(
        weather_data['date'].apply(datetime.datetime.utcfromtimestamp))
    return weather_data


def pipeline_function():
    '''
    return: dataframe for AQI and associated data between current date - 91 to current date - 1
    '''
    return GenerateWeatherDataFromJson(FetchDataBetweenDates(*GenerateHistoricalDates()))

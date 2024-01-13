import os

urls = {
    "weather_data": "http://api.openweathermap.org/data/2.5/air_pollution/history",
    "mongo_db_cluster_collection": "mongodb+srv://%s:%s@aipred.uz3wuo0.mongodb.net/?retryWrites=true&w=majority"
}

weather_params = {
    "lon": 78.4744,
    "lat": 17.3753,
    "start": None,
    "end": None,
    "appid": None
}

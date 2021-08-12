import json
import pickle5 as pickle
import joblib
import numpy as np

__locations = None
__data_columns = None
__model = None

def load_saved_artifacts():
    print("Loading server artifact ... Start")
    global __data_columns
    global __locations
    global __model

    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data-columns']
        __locations = __data_columns[3:]

    __model = joblib.load("artifacts/banglore_home_prices_model.joblib")
    print("Loading saved artifact .. Done")

def get_location_names():
    return __locations

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0], 2)

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st phase jp nagar', 1000, 2, 2))
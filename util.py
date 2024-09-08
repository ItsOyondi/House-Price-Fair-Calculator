import numpy as np
import warnings
import joblib
import json
from Regressors import RidgeRegression
warnings.filterwarnings('ignore')

__zipcodes = None
__model = None

def minmax_scaler(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_values = (data - min_val) / (max_val - min_val)
    return scaled_values
def scale_single_pred_value(value, min_val, max_val):
    scaled_value = (value - min_val) / (max_val - min_val)
    return scaled_value
def MAPE(y,y_hat):
  return round(np.mean(np.abs((y-y_hat)/y))*100,2)
def inverse_transform(column):
    return 1 / np.where(column != 0, column, 1)

def predict_house_price(zipcode,bathrooms, sqrt_ft, fireplaces, house_age, has_Dishwasher, has_Oven, has_Refrigerator, has_Freezer, has_Microwave, has_Countertops, has_Pantry, has_Others_appliances):
    
    try:
        loc_index = __zipcodes.index(zipcode)
    except:
        loc_index = -1

    bath_min = 1
    bath_max = 36

    sqrt_ft_min = 1484
    sqrt_ft_max = 5709

    fire_min = 0
    fire_max = 9

    bathrooms = scale_single_pred_value(bathrooms, bath_min, bath_max)
    sqrt_ft = scale_single_pred_value(sqrt_ft, sqrt_ft_min, sqrt_ft_max)
    fireplaces = scale_single_pred_value(fireplaces, fire_min, fire_max)
    house_age = inverse_transform(house_age)

    x = np.zeros(len(__zipcodes)+12) #Added 12 to make to the number of required features

    x[0] = bathrooms
    x[1] = sqrt_ft
    x[2] = fireplaces
    x[3] = house_age
    x[4] = has_Dishwasher
    x[5] = has_Oven
    x[6] = has_Refrigerator
    x[7] = has_Freezer
    x[8] = has_Microwave
    x[9] = has_Countertops
    x[10] = has_Pantry
    x[11] = has_Others_appliances

    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def get_house_predictions(__zipcodes, bathrooms, sqrt_ft, fireplaces, house_age, has_Dishwasher, has_Oven, has_Refrigerator, has_Freezer, has_Microwave, has_Countertops, has_Pantry, has_Others_appliances):
  zip_location_with_predicted_price = {}
  for zipcode in __zipcodes:
    predicted = round(predict_house_price(zipcode, bathrooms, sqrt_ft, fireplaces, house_age, has_Dishwasher, has_Oven, has_Refrigerator, has_Freezer, has_Microwave, has_Countertops, has_Pantry, has_Others_appliances),2)
    zip_location_with_predicted_price[zipcode] = predicted
  return zip_location_with_predicted_price
def get_min_max_from_dict(data_dict):
    min_zip = min(data_dict, key=data_dict.get)
    max_zip = max(data_dict, key=data_dict.get)
    # print(f"The cheapest house with selected features is located at : {min_zip}, Price is: {data_dict[min_zip]}")
    # print(f"The most expensive house with selected features is located at : {max_zip}, Price is: {data_dict[max_zip]}")
    return {min_zip: data_dict[min_zip]}, {max_zip: data_dict[max_zip]}

def load_artifacts():
    global __zipcodes
    global __model

    print("Loading Artifacts ... started")
    with open('artifacts/zipcodes.json', 'r') as f:
            __zipcodes = json.load(f)['zipcodes']
    __model = joblib.load('artifacts/HousePriceDecisionModel.model')

    print("Artifacts loading ... completed")

if __name__ == '__main__':
     load_artifacts()
     print(predict_house_price(85605, 10, 2000, 3,20, 1,1,1,1,1,1,1,1))
     print("----------------")
     print(get_min_max_from_dict(get_house_predictions(__zipcodes, 10, 2000, 3,20, 1,1,1,1,1,1,1,1)))
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import util
from Regressors import RidgeRegression, KNNClassifier

app = Flask(__name__)



@app.route('/predict_prices', methods=['GET', 'POST'])
def predict_prices():
    try:
        if request.method == 'POST':
            # Get form data
            zipcode = request.form.get('zipcode')
            bathrooms = float(request.form.get('bathrooms'))
            sqrt_ft = float(request.form.get('sqrt_ft'))
            latitude = request.form.get('latitude')
            longitude = request.form.get('longitude')

            try:
                latitude = float(latitude)
                longitude = float(longitude)
            except ValueError:
                return "Invalid latitude or longitude input."
            fireplaces = float(request.form.get('fireplaces'))
            house_age = float(request.form.get('house_age'))
            has_Dishwasher = 1 if request.form.get('has_Dishwasher') else 0
            has_Oven = 1 if request.form.get('has_Oven') else 0
            has_Refrigerator = 1 if request.form.get('has_Refrigerator') else 0
            has_Freezer = 1 if request.form.get('has_Freezer') else 0
            has_Microwave = 1 if request.form.get('has_Microwave') else 0
            has_Countertops = 1 if request.form.get('has_Countertops') else 0
            has_Pantry = 1 if request.form.get('has_Pantry') else 0
            has_Others_appliances = 1 if request.form.get('has_Others_appliances') else 0
            category = util.get_class(longitude, latitude)

            # Make a prediction
            prediction =  util.predict_house_price(zipcode, bathrooms, sqrt_ft, fireplaces, house_age,
                                  has_Dishwasher, has_Oven, has_Refrigerator, has_Freezer,
                                  has_Microwave, has_Countertops, has_Pantry, has_Others_appliances, category)
            min_price, max_price = util.get_min_max_from_dict(util.get_house_predictions(util.__zipcodes,bathrooms, sqrt_ft, fireplaces, house_age,
                                  has_Dishwasher, has_Oven, has_Refrigerator, has_Freezer,
                                  has_Microwave, has_Countertops, has_Pantry, has_Others_appliances, category))

            # Return the prediction result
            # Extract zip codes and values for max and min prices
            max_zipcode, max_value = list(max_price.items())[0]
            min_zipcode, min_value = list(min_price.items())[0]

            # Return the prediction result along with min and max values
            return render_template('results.html', 
                                selected_zip = zipcode,
                                prediction=prediction,
                                min_zipcode=min_zipcode, 
                                min_value=min_value,
                                max_zipcode=max_zipcode, 
                                max_value=max_value
                                )

        return render_template('app.html', zipcodes=util.__zipcodes)
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == '__main__':
    print("Server starting to render ...")
    util.load_artifacts()
    app.run(debug=True)

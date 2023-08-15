from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load('static/model_rf.pkl')

@app.route('/predict', methods=['POST'])
def predict():
  try:
    # Get JSON data from the request
    data = request.json

    # Extract features from the data
    # Ensure that the features are in the same order as used for training
    features = [data['OCC_YEAR'], data['OCC_DOW'], data['OCC_DAY'],data['OCC_DOY'],data['OCC_HOUR'],data['REPORT_YEAR'],data['REPORT_MONTH'],data['REPORT_DOW'],data['REPORT_DAY'],data['REPORT_DOY'],data['REPORT_HOUR'],data['DIVISION'],data['PREMISES_TYPE'],data['BIKE_MAKE'],data['BIKE_MODEL'],data['BIKE_TYPE'],data['BIKE_SPEED'],data['BIKE_COLOUR'],data['BIKE_COST'],data['LOCATION_TYPE']]

    # Predict using the loaded model
    prediction = model.predict([features])[0]

    # Return the prediction as JSON response
    return jsonify({"prediction": str(prediction)})

  except Exception as e:
    return jsonify({"error": str(e)})


if __name__ == '__main__':
  app.run(debug=True)

from flask import Flask, render_template, jsonify
import pandas as pd
import pickle
from flask import request
import numpy as np

app = Flask(__name__)

# Load model and encoders
with open("model/plastic_weight_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/region_encoder.pkl", "rb") as f:
    region_encoder = pickle.load(f)
with open("model/plastic_type_encoder.pkl", "rb") as f:
    plastic_type_encoder = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html', active_page='index')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html', active_page='prediction')

@app.route('/data')
def data():
    df = pd.read_csv('data/ocean_plastic_pollution_data_2015.csv')
    return df.to_json(orient='records')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        region = data['region']
        plastic_type = data['plastic_type']
        depth = float(data['depth'])
        year = int(data['year'])

        # Encode inputs
        region_code = region_encoder.transform([region])[0]
        plastic_type_code = plastic_type_encoder.transform([plastic_type])[0]

        # Prepare input array
        input_array = np.array([[depth, region_code, plastic_type_code, year]])

        # Predict
        prediction = model.predict(input_array)[0]

        return jsonify({'prediction': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
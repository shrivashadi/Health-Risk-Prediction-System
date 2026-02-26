from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    # Get data from user
    data = request.get_json()
    
    # Convert into dataframe
    input_df = pd.DataFrame([data])
    
    # Load trained model
    with open('model/model.pkl', 'rb') as obj:
        model = pickle.load(obj)
    
    # Load columns
    with open('model/columns.pkl', 'rb') as obj:
        model_columns = pickle.load(obj)
    
    # Align columns with training data
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)
    prediction = round(float(prediction[0]), 2)
    
    # Return response
    response = {"Prediction": prediction}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
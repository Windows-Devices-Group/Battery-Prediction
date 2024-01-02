#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the serialized model
with open('gbrt_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return 'ML Model Deployment with Flask on PythonAnywhere'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)  # Adjust this based on your model's input

    # Make predictions using the loaded model
    prediction = model.predict(X_test)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run()


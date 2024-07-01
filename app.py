import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

app=Flask(__name__) # Starting point of the application from where it will run

# Load the model
reg_model=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html') # Whenever we hit the flask app it will first redirect to the home.html page

@app.route('/predict_api',methods=['POST']) 
def predict_api():
    data=request.json['data']
    print(data)
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = reg_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input =  scalar.transform(np.array(data).reshape(1,-1))
    output = reg_model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The house price is {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)
    



 

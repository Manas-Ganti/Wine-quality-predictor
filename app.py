from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

from pyunpack import Archive
Archive('random_forest_winepred.rar').extractall('')

model = pickle.load(open('random_forest_winepred.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html') 
    
standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == "POST":
        wine_type = int(request.form['wine_type']) 
        fixed_acidity = float(request.form['fixed_acidity'])
        volatile_acidity = float(request.form["volatile_acidity"])
        citric_acid = float(request.form["citric_acid"])
        residual_sugar = float(request.form["residual_sugar"])
        chlorides = float(request.form["chlorides"])
        free_so2 = float(request.form["free_so2"])
        total_so2 = float(request.form["total_so2"])
        density = float(request.form["density"])
        pH = float(request.form["pH"])
        sulphates = float(request.form["sulphates"])
        alcohol = float(request.form["alcohol"])

        prediction=model.predict([[wine_type,fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_so2,total_so2,density,pH,sulphates,alcohol]])
        output=round(prediction[0],2)


        return render_template("index.html",prediction_text = "Wine Quality is {}" .format(output))    


if __name__=="__main__":
    app.run(debug=True)
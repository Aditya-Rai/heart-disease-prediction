# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:40:12 2021

@author: Aditya

"""

from flask import Flask , render_template , request ,jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route("/" )
def Hello():
    return render_template("predictor.html")


@app.route("/prediction" , methods = ["POST"])
def predict():
    
    if request.method == 'POST':
        age=float(request.form['age'])
                 
        sex = request.form['Sex']
         
        restingbp = float(request.form['resitngbp'])

        cholesterol = float(request.form['cholesterol'])
        
        fastingBS = float(request.form['fastingBS'])
        
        maxhr = float(request.form['maxhr'])
    
        oldpeak = float(request.form['oldpeak'])
        
        ChestPainType = request.form['ChestPain']
    
        ST_Slope = request.form['ST_Slope']
        
        RestingECG = request.form["RestingECG"]
        
        ExerciseAngina = request.form['ExerciseAngina']
        
        age = np.round(((age - 53.848774)/9.427477),2)
        
        restingbp = np.round(((np.log(restingbp + 1) - 132.874659)/18.06800),2)
        
        cholesterol = np.round(((cholesterol - 203.227520)/108.328198),2)
        
        fastingBS = np.round(((fastingBS - 0.228883)/0.4228151),2)
        
        maxhr = np.round(((maxhr - 136.377384)/25.32449),2)
        
        oldpeak = np.round(((oldpeak - 0.865123)/1.065989),2)
        
 
        
        
         

        Sex_M = 0
        if(sex == "male"):
            Sex_M = 1
            
            
        ChestPainType_ATA = 0
        ChestPainType_NAP = 0
        ChestPainType_TA = 0
        
        
        if ChestPainType == "ata":
            ChestPainType_ATA = 1
        elif ChestPainType == "nap":
            ChestPainType_NAP = 1
        elif ChestPainType == "ta":
            ChestPainType_TA = 1
        
        RestingECG_Normal = 0
        RestingECG_ST = 0
        
        if RestingECG == "normal":
            RestingECG_Normal = 1
        elif RestingECG == "st":
            RestingECG_ST = 1
            
        
        ExerciseAngina_Y = 0
        
        if ExerciseAngina == "yes":
            ExerciseAngina_Y = 1
            
        ST_Slope_Flat = 0
        ST_Slope_Up = 0
        
        if ST_Slope == "flat":
            ST_Slope_Flat = 1
        elif ST_Slope == "up":
            ST_Slope_Up = 1
                
            
        
       
        x = [age,restingbp,cholesterol,fastingBS,maxhr,oldpeak,Sex_M,ChestPainType_ATA,
             ChestPainType_NAP,ChestPainType_TA,RestingECG_Normal,RestingECG_ST,
             ExerciseAngina_Y,ST_Slope_Flat,ST_Slope_Up]
        
        
        
        prediction=model.predict_proba([x])
        output=(np.round(prediction[0][1],2)) * 100
        output = np.round(output,2)
        
        if output > 50.0:
            return render_template("danger.html", prediction_text ="{}".format(output) )
        else:
            return render_template("safe.html",prediction_text = "{}".format(output))
        
        return render_template('predictor.html',prediction_text=
                               "The patient has {}% chance of Heart Disease".format(output))
    
        
    else:
        return render_template('predictor.html')    

if __name__ == "__main__":
    app.run(debug=True)
    

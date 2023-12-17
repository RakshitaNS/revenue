from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.components.src_.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Id=request.form.get('Id'),
            Name=request.form.get('Name'),
            Franchise=request.form.get('Franchise'),
            Category=request.form.get('Category'),
            City=request.form.get('City'),
            No_Of_Item=request.form.get('No_Of_Item'),
            Order_Placed=float(request.form.get('Order_Placed'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.debug = True
    app.run(host="0.0.0.0")        
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np

app = FastAPI()

std = joblib.load('Standard_cardio.pkl')
model = joblib.load('cardio.pkl')

class Features(BaseModel):
    Age:float
    Height_in_cm:float
    Weight_in_kg:float
    High_BP:float
    Low_BP:float
    Gender:float
    Glucose:float
    Smoke:float
    Alcohol:float
    Active:float
    Cholestrol:float

@app.get('/')
def home():
    return 'Health Prediction'

@app.post('/predict')
async def predict(data: Features):
    data = data.dict()
    Age = data['Age']
    Height_in_cm = data['Height_in_cm']
    Weight_in_kg = data['Weight_in_kg']
    High_BP = data['High_BP']
    Low_BP = data['Low_BP']
    Gender = data['Gender']
    Glucose = data['Glucose']
    Smoke = data['Smoke']
    Alcohol = data['Alcohol']
    Active = data['Active']
    Cholestrol = data['Cholestrol']
    
    s = std.transform(np.array([[Age, Height_in_cm, Weight_in_kg, High_BP, Low_BP]]))
    pred = model.predict(np.concatenate((s, [[Gender, Glucose, Smoke, Alcohol, Active, Cholestrol]]),axis=1))
    
    if pred == 0:
        return 'Result:- The model has predicted that you will not suffer from any cardic arresst but you should take care of your self.'
    else:
        return 'Result:- You should consult with doctor, The model has predicted that you will suffer form cardic arrest.'



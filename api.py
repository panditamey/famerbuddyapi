from fastapi import FastAPI,UploadFile,File
from pydantic import BaseModel
import pickle
import json
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import os

CHUNK_SIZE = 1024

app = FastAPI()

class crop_recommend_input(BaseModel):
    N : int
    P : int
    K : int
    temperature : float
    humidity : float
    ph : float
    rainfall : float

class crop_yield_input(BaseModel):
    State_Name : str
    District_Name : str
    Season : str
    Crop : str
    Area : float
    Production : float

crop_recommend_ml = pickle.load(open('models/CropRecommendationSystem','rb'))
crop_yield_ml = pickle.load(open('models/CropYieldPrediction.pkl','rb'))
crop_disease_ml=load_model('models/CropDiseaseDetection.h5')

@app.post('/croprecommend')
def croprecommend(input_parameters : crop_recommend_input):

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    N = input_dictionary['N']
    P = input_dictionary['P']
    K = input_dictionary['K']
    temperature = input_dictionary['temperature']
    humidity = input_dictionary['humidity']
    ph = input_dictionary['ph']
    rainfall = input_dictionary['rainfall']
    input_list = [N, P, K, temperature, humidity, ph, rainfall]
    prediction = crop_recommend_ml.predict([input_list])
    print(prediction[0])
    return {"crop":str(prediction[0])}

@app.post('/cropyield')
def cropyield(input_parameters : crop_yield_input):

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    State_Name = input_dictionary['State_Name']
    District_Name = input_dictionary['District_Name']
    Season = input_dictionary['Season']
    Crop = input_dictionary['Crop']
    Area = input_dictionary['Area']
    Production = input_dictionary['Production']
    input_list = [State_Name, District_Name, Season, Crop, Area, Production]
    # df = pd.DataFrame([['Chhattisgarh',	'BEMETARA',	'Rabi'	,'Potato',	3.0	,20.0]], columns=['State_Name',	'District_Name',	'Season',	'Crop',	'Area'	,'Production'])
    df = pd.DataFrame([input_list], columns=['State_Name',	'District_Name',	'Season',	'Crop',	'Area'	,'Production'])
    prediction = crop_yield_ml.predict(df)
    return {"yield":float(prediction[0])}

@app.post('/cropdisease')
async def cropdisease(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    classes = ['Potato___Early_blight', 'Tomato_healthy', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato_Septoria_leaf_spot', 'Tomato_Bacterial_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Pepper__bell___healthy', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Potato___healthy', 'Tomato_Leaf_Mold', 'Potato___Late_blight', 'Pepper__bell___Bacterial_spot']
    img=image.load_img(str(file.filename),target_size=(224,224))
    x=image.img_to_array(img)
    x=x/255
    x=np.expand_dims(x,axis=0)
    img_data=preprocess_input(x)
    prediction = crop_disease_ml.predict(img_data)
    predictions = list(prediction[0])
    max_num = max(predictions)
    index = predictions.index(max_num)
    print(classes[index])
    os.remove(str(file.filename))
    return {"disease":classes[index]}

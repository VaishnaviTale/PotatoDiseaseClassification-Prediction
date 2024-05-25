from fastapi import FastAPI, File, UploadFile
import uvicorn
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import requests

app = FastAPI()

endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

CLASSNAME = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello"

def read_file_as_image(data) -> np.ndarray : 
    image = np.array(Image.open(io.BytesIO(data)))
    return image
     

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)


    json_data = {
        "instances" : img_batch.tolist()
    }

    requests.post(endpoint, json=json_data)
    
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASSNAME[np.argmax(prediction)]
    confidence = np.max(prediction)

    return{
        'class' : predicted_class,
        'confidence' : float(confidence)
    }



if __name__ == "__main__":
  uvicorn.run(app, host='localhost' ,port=3000)

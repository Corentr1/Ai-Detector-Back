# TODO: Import your package, replace this by explicit imports of what you need

#from packagename.main import predict
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from ai_detector.model_logic.model import initialize_model, compile_model, train_model, evaluate_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
@app.get("/predict")
def get_predict(input_one: float,
            input_two: float):

    prediction = float(input_one) + float(input_two)

    return {
        'prediction': prediction
    }

# Receiving image from frontend
@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    #nparr = np.fromstring(contents, np.uint8)
    #cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
    tensor_image = tf.io.decode_image(contents)

    ### Do cool stuff with your image.... For example face detection
    return {
        'size of the array':len(tensor_image.shape)
        }

# Use a first version of the model
@app.post('/first_model')
async def first_model(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    tensor_image = tf.io.decode_image(contents,
                                      channels=3)
    tensor_image = tf.image.resize(tensor_image,
                                   size=[32, 32]
                                   )

    ### create the model
    model = initialize_model(
         input_shape=tuple(tensor_image.shape)
      )
    model = compile_model(model)
    X = np.expand_dims(np.array(tensor_image), axis=0)
    y = np.array([1])
    model, history=train_model(
         model,
         X,
         y)

    score = model.evaluate(X, y)


    return score

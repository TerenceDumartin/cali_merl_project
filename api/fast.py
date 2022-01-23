from fastapi import FastAPI, Response, Request, Header, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from tensorflow.keras import models
import numpy as np
import cv2
import os
import shutil
import time
from collections import deque

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Instanciate cache object to mem store models
cache_models = {}

labels_list = ['Inspect_Shelf', 'Inspect_Product', 'Hand_in_Shelf', 'Reach_to_Shelf', 'Retract_from_Shelf']


def check_extension(filename):
    ALLOWED_EXTENSION = ["mp4"]
    # Extract extension
    extension = filename.split(".")[-1:][0].lower()
    if extension not in ALLOWED_EXTENSION:
        return False
    else:
        return True


@app.on_event("startup")
async def startup_event():
    '''
    On api startup, load and store model in memory
    '''
    print("load model ...")
    model_0 = models.load_model('cali_merl_project/models/VGG16___LOSS_08___ACCURACY_97.h5')
    cache_models["model_0"] = model_0
    print("models are ready ...")


# Return only the pred and proba
@app.post("/predict")
async def predict_handler(response: Response, inputImage: UploadFile = File(...)):
    '''
    Check extension
    '''
    check = check_extension(inputImage.filename)

    if check == False:
        response_payload = {
                "status": "error",
                "message": "Input file format not valid - Only .mp4"
                }
        response.status_code = 400
        return response

    '''
    Copy File as buffer in tmp
    '''
    temp_image = str(int(time.time())) + "_" + inputImage.filename
    with open(temp_image, "wb") as buffer:
        shutil.copyfileobj(inputImage.file, buffer)

    await inputImage.close()
    buffer.close()

    '''
    Predict !
    '''
    video_reader_pred = cv2.VideoCapture(temp_image)
    total_number_frames = int(video_reader_pred.get(cv2.CAP_PROP_FRAME_COUNT))
    pred_deque = deque(maxlen=total_number_frames)

    while True:
        status, frame = video_reader_pred.read()

        if not status:
            break
        resized_frame = cv2.resize(frame, (128, 128))
        normalized_frame = resized_frame / 255

        model_0 = cache_models["model_0"]

        # prediction
        pred = model_0.predict(np.expand_dims(normalized_frame, axis=0))[0]

        pred_deque.append(pred)

        if len(pred_deque) == total_number_frames:
            # Deque
            pred_np = np.array(pred_deque)
            pred_averaged = pred_np.mean(axis=0)
            predicted_label = np.argmax(pred_averaged)

            predicted_label_name = labels_list[predicted_label.tolist()]
            predict_proba = pred_averaged[predicted_label]

            response_payload = {
                    'prediction': predicted_label_name,
                    'proba': predict_proba.tolist(),
                    }

            response.status_code = 202
            response.headers["Content-Type"] = "application/json"

            if os.path.exists(temp_image):
                os.remove(temp_image)

            return response_payload

    if os.path.exists(temp_image):
        os.remove(temp_image)


# Return the file with the prediction overlay
@app.post("/predict_w_clip/")
async def predict_handler(response: Response, inputImage: UploadFile = File(...)):
    '''
    Check extension
    '''
    check = check_extension(inputImage.filename)

    if check == False:
        response_payload = {
                "status": "error",
                "message": "Input file format not valid - Only .mp4"
                }
        response.status_code = 400
        return response

    temp_image = str(int(time.time())) + "_" + inputImage.filename
    with open(temp_image, "wb") as buffer:
        shutil.copyfileobj(inputImage.file, buffer)

    await inputImage.close()
    buffer.close()

    video_reader_pred = cv2.VideoCapture(temp_image)
    total_number_frames = int(video_reader_pred.get(cv2.CAP_PROP_FRAME_COUNT))
    pred_deque = deque(maxlen=total_number_frames)

    original_video_width = int(video_reader_pred.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader_pred.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(f'layered_{temp_image}', cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 30, (original_video_width, original_video_height))

    while True:
        status, frame = video_reader_pred.read()

        if not status:
            break
        resized_frame = cv2.resize(frame, (128, 128))
        normalized_frame = resized_frame / 255

        model_0 = cache_models["model_0"]

        # prediction
        pred = model_0.predict(np.expand_dims(normalized_frame, axis=0))[0]

        pred_deque.append(pred)

        if len(pred_deque) == total_number_frames:
            # Deque
            pred_np = np.array(pred_deque)
            pred_averaged = pred_np.mean(axis=0)
            predicted_label = np.argmax(pred_averaged)

            predicted_label_name = labels_list[predicted_label.tolist()]
            predict_proba = pred_averaged[predicted_label]
            text_2_plot = f'{str(predicted_label_name)} - {str(round(predict_proba*100,2))} %'

            # Add Label on top of the clips
            x, y, w, h = 0, 0, 350, 70
            cv2.rectangle(frame, (x, x), (x + w, y + h), (0, 0, 128), -1)
            cv2.putText(frame, text_2_plot, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        video_writer.write(frame)

    video_reader_pred.release()
    video_writer.release()

    if os.path.exists(temp_image):
        os.remove(temp_image)

    response.status_code = 200
    FRes = FileResponse(f'layered_{temp_image}', media_type="video/mp4")

    return FRes

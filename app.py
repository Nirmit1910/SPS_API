import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify,request
import xgboost
from flask_cors import CORS
from PIL import Image
import sys
import joblib
from io import BytesIO
import mediapipe as mp

app=Flask(__name__)
CORS(app)

classifier = joblib.load(("Predictor.pkl"))

class HandDetector():
    def __init__(self,
                 maxHands = 2,
                 detectionCon = 0.5,
                 trackCon = 0.5):

        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            min_tracking_confidence=self.trackCon,
            min_detection_confidence=self.detectionCon,
            max_num_hands=self.maxHands
        )

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img):
        results = self.hands.process(img)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)

                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return results

    def get_landmarks(self, img):
        results = self.hands.process(img)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lm_dict = {}
                for id, Lm in enumerate(handLms.landmark):
                    lm_dict[f"{id}_x"] = Lm.x
                    lm_dict[f"{id}_y"] = Lm.y
                lms = pd.Series(lm_dict)
                lms = lms.to_numpy()
                lms = np.expand_dims(lms, axis = 0)
            return lms
        else:
            return None


        

    def get_result_as_dict(self, img, classifier):
        lms = self.get_landmarks(img)
        predicted_as_dict = {"Paper": 0,"Scissor": 0, "Stone": 0}
        prediction_arr = None
        if (lms is not None):
            prediction_arr = classifier.predict_proba(lms)
            pred = np.argmax((prediction_arr), axis=-1)
            if(prediction_arr[0][pred] > 0.8):
                if(pred == 2):
                    predicted_as_dict['Stone'] = 1
                elif(pred == 1):
                    predicted_as_dict['Scissor'] = 1
                elif(pred == 0):
                    predicted_as_dict['Paper'] = 1
        return predicted_as_dict


@app.route('/')
def index():
    result={
        "message":"api running"
    }
    return jsonify(result)

@app.route('/getResult',methods=['POST'])
def get_result_as_dict():
        hand_detector = HandDetector()
        if request.method == 'POST':
            image=request.json
            # print(type(image["image"]))
            image=np.array(image["image"])
            image=image.astype('uint8')
            predicted_data = hand_detector.get_result_as_dict(image, classifier)

            # print(predicted_data)
            
            return predicted_data

if __name__ == '__main__':
    app.run(debug=True)
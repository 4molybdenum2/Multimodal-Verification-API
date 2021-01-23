import numpy as np
from tensorflow.keras.models import load_model

def predict_person(result):

    image1 , image2 , audio1 , audio2 = result
    model = load_model('./ml_model/model.h5')
    preds = model.predict([np.asarray(image1) , np.asarray(audio1) , np.asarray(image2) , np.asarray(audio2)])
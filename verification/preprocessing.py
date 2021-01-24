import os
import numpy as np
import cv2
import wave
import pylab
import matplotlib .pyplot as plt
from scipy import signal
from scipy.io import wavfile
from tensorflow.keras.models import load_model
from django.conf import settings
from PIL import Image
from django.core.files.storage import default_storage

from django.core.files.base import ContentFile

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

def image_processing(path,xyz):
    array_bgr = cv2.imread(path)
    array = cv2.cvtColor(array_bgr , cv2.COLOR_BGR2RGB)
    img= cv2.resize(array , (IMAGE_HEIGHT,IMAGE_WIDTH))
    img= np.array(img).reshape(-1,IMAGE_HEIGHT,IMAGE_WIDTH,3)
    if xyz==1:
        plt.imshow(array)
        plt.show()
    return img

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

def audio_to_spectrogram(file_name , save_path):
    ##############################
    
    sound_info, frame_rate = get_wav_info(file_name)
    _ , _ ,_ , X = plt.specgram(sound_info, Fs=frame_rate)
    # print(X)
    # im = Image.fromarray(X[0])
    opencvImage = cv2.cvtColor(np.array(X), cv2.COLOR_RGB2BGR)


    ##############################
    # array_bgr = cv2.imread(os.path.join(settings.MEDIA_ROOT,save_path))
    # array = cv2.cvtColor(array_bgr , cv2.COLOR_BGR2RGB)
    img= cv2.resize(opencvImage , (IMAGE_HEIGHT,IMAGE_WIDTH))
    img= np.array(img).reshape(-1,IMAGE_HEIGHT,IMAGE_WIDTH,3)
    
    print(img)
    return img

def preprocessing_inputs(file_image1 , file_image2 , file_audio1 , file_audio2):
    image1 = image_processing(file_image1,0)
    image2 = image_processing(file_image2,0)
    audio1 = audio_to_spectrogram(file_audio1 , 'tmp/audio_image/audio1_image.png')
    audio2 = audio_to_spectrogram(file_audio2 , 'tmp/audio_image/audio2_image.png')

    model = load_model('verification/ml_model/model.h5')
    preds = model.predict([np.asarray(image1) , np.asarray(audio1) , np.asarray(image2) , np.asarray(audio2)])

    return preds

    
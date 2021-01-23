import numpy as np
import cv2
from scipy import signal
from scipy.io import wavfile
from tensorflow.keras.models import load_model
import librosa

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

def audio_to_spectrogram(file_name):
    signal, sr = librosa.load(file_name, sr=44100)
    n_fft = 2048
    hop_length = 512
    stft = librosa.core.stft(signal , hop_length=hop_length ,n_fft=n_fft )
    spectrogram = np.abs(stft)
    return np.resize(spectrogram , (1,128,128,3))
    
def preprocessing_inputs(file_image1 , file_image2 , file_audio1 , file_audio2):
    image1 = image_processing(file_image1,0)
    image2 = image_processing(file_image2,0)
    audio1 = audio_to_spectrogram(file_audio1)
    audio2 = audio_to_spectrogram(file_audio2)

    model = load_model('verification/ml_model/model.h5')
    preds = model.predict([np.asarray(image1) , np.asarray(audio1) , np.asarray(image2) , np.asarray(audio2)])

    return preds

    
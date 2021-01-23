import numpy as np
import cv2
from scipy import signal
from scipy.io import wavfile

# model = load_model('./models/MultimodalModel')
def image_processing(path,xyz):
    array_bgr = cv2.imread(path)
    array = cv2.cvtColor(array_bgr , cv2.COLOR_BGR2RGB)
    img= cv2.resize(array , (IMAGE_HEIGHT,IMAGE_WIDTH))
    img= np.array(img).reshape(-1,IMAGE_HEIGHT,IMAGE_WIDTH,3)
    if xyz==1:
        plt.imshow(array)
        plt.show()
    return img

def audio_to_spectrogram(file):
    sample_rate, samples = wavfile.read(file)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    return spectrogram
    
def preprocessing_inputs(file_image1 , file_image2 , file_audio1 , file_audio2):
    image1 = image_processing(file_image1,0)
    image2 = image_processing(file_image2,0)
    audio1_image = audio_to_spectrogram(file_audio1)
    audio2_image = audio_to_spectrogram(file_audio2)
    audio1 = image_processing(audio1_image)
    audio2 = image_processing(audio2_image)

    return image1 , image2 , audio1 , audio2
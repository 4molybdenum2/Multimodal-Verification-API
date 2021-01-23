from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.decorators import api_view

from .preprocessing import preprocessing_inputs
from .predict import predict_person

@api_view(['GET'])
def api_overview(request):
    # Also Include Details or redirect to a page
    api_urls = {
        '/upload : upload a file to the server',
    }
    return Response(api_urls)

@api_view(["POST"])
def predict_person(request):
    try:
        file_image1 = request.data.get('image1',None)
        file_image2 = request.data.get('image2',None)
        file_audio1 = request.data.get('audio1',None)
        file_audio2 = request.data.get('audio2',None)
        fields = [file_image1,file_image2,file_audio1,file_audio2]
        if not None in fields:
            # Datapreprocessing
            # Converting the images to numpy arrays and converting the audio files to spectogram and then into numpy arrays.
            image1 , image2 , audio1 , audio2 = preprocessing_inputs(file_image1 , file_image2 , file_audio1 , file_audio2)
            result = (image1 , image2 , audio1 , audio2)
            #Passing data to model & loading the model from disks
            preds = predict_person(result)
            
            if preds[0] < 0.6:
                predictions = {
                'error' : '0',
                'message' : 'Successful',
                'prediction' : 'Different Person with'+str(preds[0]*100)+'% probability'
            }
            else:
                predictions = {
                'error' : '0',
                'message' : 'Successful',
                'prediction' : 'Same Person with'+str(preds[0]*100)+'% probability'
            }
            
        else:
            predictions = {
                'error' : '1',
                'message': 'Invalid Parameters'                
            }
    except Exception as e:
        predictions = {
            'error' : '2',
            "message": str(e)
        }
    
    return Response(predictions)

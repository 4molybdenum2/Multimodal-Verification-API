import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.decorators import api_view

from .preprocessing import preprocessing_inputs

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
        ##########################################
        file_image1 = request.data.get('image1',None)
        file_image2 = request.data.get('image2',None)
        file_audio1 = request.data.get('audio1',None)
        file_audio2 = request.data.get('audio2',None)
        ###########################################

        path_image1 = default_storage.save('tmp/image1.png', ContentFile(file_image1.read()))
        path_image2 = default_storage.save('tmp/image2.png', ContentFile(file_image2.read()))
        path_audio1 = default_storage.save('tmp/audio1.wav', ContentFile(file_audio1.read()))
        path_audio2 = default_storage.save('tmp/audio2.wav', ContentFile(file_audio2.read()))

        ###########################################
        tmp_file_image1 = os.path.join(settings.MEDIA_ROOT, path_image1)
        tmp_file_image2 = os.path.join(settings.MEDIA_ROOT, path_image2)
        tmp_file_audio1 = os.path.join(settings.MEDIA_ROOT, path_audio1)
        tmp_file_audio2 = os.path.join(settings.MEDIA_ROOT, path_audio2)

        fields = [tmp_file_image1,tmp_file_image2,tmp_file_audio1,tmp_file_audio2]

        if not None in fields:
            # Datapreprocessing
            # Converting the images to numpy arrays and converting the audio files to spectogram and then into numpy arrays.
            preds= preprocessing_inputs(tmp_file_image1 , tmp_file_image2 , tmp_file_audio1 , tmp_file_audio2)
            
            # os.remove(tmp_file_image1)
            # os.remove(tmp_file_image2)
            # os.remove(tmp_file_audio1)
            # os.remove(tmp_file_audio2)

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

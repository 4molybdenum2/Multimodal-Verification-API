from django.urls import path
from .views import *

urlpatterns = [
    path('', api_overview),
    path('upload/', predict_person)
]

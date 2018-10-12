
from .lib import CurveFit
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from django.core import serializers
from rest_framework.views import APIView
from django.http import HttpResponse

def index(request):

    request.session['my_list'] = CurveFit.startApp(request.POST['datastring'],request.POST['dataLength'])
    return HttpResponse(request.session['my_list'], content_type="application/json")


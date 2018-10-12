from django.conf.urls import url
from django.contrib import admin
from django.urls import path, include
from rest_framework.urlpatterns import format_suffix_patterns

from CurveFitting import views as FIT

urlpatterns = (
    path('admin/', admin.site.urls), # admin site

)


urlpatterns += (
    # urls for Django Rest Framework API
    url(r'^api/stats/', FIT.index, name='DataStats'),


)
urlpatterns = format_suffix_patterns(urlpatterns)




import os

from django.urls import re_path, path

from . import views
from django.conf.urls import url
from django.views.static import serve

try:
    request_handler = views.RequestHandler(config_path=os.environ['CONFIG_PATH'])
except KeyError as e:
    raise ValueError('Please set `CONFIG_PATH` environmental vars!') from e

urlpatterns = [
    url(r'^$', request_handler.index, name='index'),
    url(r'^upload/', request_handler.upload, name='upload'),
    path(r'result/<config_name>/', request_handler.result, name='result_param'),
    url(r'^result/', request_handler.result, name='result'),
    url(r'^progress/', request_handler.get_progress, name='progress'),
    url(r'^vdo/(?P<path>.*)$', serve, {'document_root': './', 'show_indexes': True}),
]

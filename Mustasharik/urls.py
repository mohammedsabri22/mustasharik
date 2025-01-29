"""
Definition of urls for Mustasharik.
"""

from datetime import datetime
from django.urls import path
from app import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('contract/', views.contract, name='contract'),
    path('chat-with-gpt/', views.chat_view, name='chat-with-gpt'),
  
   
]
urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

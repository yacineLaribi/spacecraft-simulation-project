from django.conf import settings
from django.conf.urls.static import static
from django.urls import path 
from . import views 

urlpatterns = [
    path('', views.index , name="home"),
    path('simulation/', views.simulation , name="simulation"),
    path('about/', views.about , name="about"),
     
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

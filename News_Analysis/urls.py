from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = "News_Analysis"
urlpatterns = [
                  path('', views.home, name='home'),
                  path('images/', views.images, name='images'),
                  path('text_files/', views.text_files, name='text_files'),
                  # path('images/details/', views.details, name='details'),
                  # path('text_files/details/', views.details, name='details')
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

from django.contrib import admin
from django.urls import path
from DigiAi import views as user_views
from mlmodel.views import FileUploadView

urlpatterns = [
    path('users/', user_views.studentApi, name='users'),
    path('users/<int:id>/', user_views.studentApi, name='user'),
    path('upload/', FileUploadView.as_view(), name='file_upload'),
    path('admin/', admin.site.urls),
]

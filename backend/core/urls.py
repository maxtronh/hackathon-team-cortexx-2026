from django.urls import path
from core.views import health_check

urlpatterns = [
    path('api/health/', health_check, name='health_check'),
]

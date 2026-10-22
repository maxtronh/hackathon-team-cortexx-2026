from django.urls import path
from core.views import health_check
from rag.views import upload_document, retrieve_passages, list_documents

urlpatterns = [
    path('api/health/', health_check, name='health_check'),
    path('api/upload/', upload_document, name='upload_document'),
    path('api/retrieve/', retrieve_passages, name='retrieve_passages'),
    path('api/documents/', list_documents, name='list_documents'),
]

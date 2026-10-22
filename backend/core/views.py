from django.http import JsonResponse

def health_check(request):
    """
    Minimal health check endpoint for preflight verification.
    """
    return JsonResponse({
        'status': 'ok',
        'service': 'document-intelligence-backend',
        'framework': 'django',
    })

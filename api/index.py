from app import app

# Vercel entry point
def handler(request):
    return app(request.environ, request.start_response)

# For Vercel compatibility
application = app
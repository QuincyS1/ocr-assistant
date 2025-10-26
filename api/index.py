import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import app

# Vercel entry point
def handler(event, context):
    return app

# Export for Vercel
app = app
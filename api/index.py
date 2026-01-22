import sys
import os

# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Vercel environment flag
os.environ['VERCEL'] = '1'

# Import the Flask app
from app import app

# Vercel expects the app to be named 'app' or 'handler'
# This file serves as the entry point for Vercel serverless functions

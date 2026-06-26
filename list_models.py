import os
import sys
from google import genai

def list_models():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set.")
        sys.exit(1)
        
    client = genai.Client(api_key=api_key)
    try:
        models = client.models.list()
        print("Available models:")
        for m in models:
            print(f"- {m.name} (Supported actions: {m.supported_generation_methods})")
    except Exception as e:
        print("Error listing models:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    list_models()

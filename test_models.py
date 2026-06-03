from google import genai
from config import GEMINI_API_KEY
client = genai.Client(api_key=GEMINI_API_KEY)
for m in client.models.list():
    if 'flash' in m.name:
        print(m.name)

from google import genai
from config import GEMINI_API_KEY
client = genai.Client(api_key=GEMINI_API_KEY)

for model_name in ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-1.5-flash']:
    try:
        response = client.models.generate_content(model=model_name, contents='Say hello')
        print(f'{model_name} SUCCESS: {response.text.strip()}')
    except Exception as e:
        print(f'{model_name} FAILED: {e}')

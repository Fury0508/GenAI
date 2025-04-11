from google import genai
from google.genai import types

client = genai.Client(api_key = 'AIzaSyBlxeJhMExzy1N_REiumSOzfbJcF6JpvK8')

response = client.models.generate_content(
    model = 'gemini-2.0-flash-001', content = 'Why is sky blue?'
)

print(response.text)
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# project = os.getenv("GOOGLE_CLOUD_PROJECT")
# location = os.getenv("GOOGLE_CLOUD_LOCATION")
# model = os.getenv("GEMINI_MODEL")

client = genai.Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION")
)

response = client.models.generate_content(
    model=os.getenv("GEMINI_MODEL"),
    contents="Explain what Google Cloud Model Garden is."
)

print(response.text)
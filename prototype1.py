import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# --- CONFIG ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")  
RAW_DIR = Path("scrap_data")
CLEANED_DIR = Path("cleansed_data")
CLEANED_DIR.mkdir(parents=True, exist_ok=True)
MODEL = os.getenv("GEMINI_MODEL")

missing = [
    name
    for name, value in (
        ("GOOGLE_CLOUD_PROJECT", PROJECT_ID),
        ("GOOGLE_CLOUD_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION")),
        ("GEMINI_MODEL", MODEL),
    )
    if not value
]
if missing:
    raise SystemExit(f"Missing env vars (check .env): {', '.join(missing)}")

# --- LLM CLIENT ---
client = genai.Client(enterprise=True, project=PROJECT_ID, location=os.getenv("GOOGLE_CLOUD_LOCATION"))

# --- CLEANING PROMPT ---
# This is where you define what "cleaning" means for your scrape data
SYSTEM_PROMPT = """
You are a data cleaning assistant. The user will give you raw scraped text from a website crawler.
Your job is to extract ONLY the main content — the actual information in the center of the page.
Remove: navigation menus, headers, footers, ads, breadcrumbs, repeated links, boilerplate text.
Keep: article body, descriptions, main text, factual content.
Return ONLY the cleaned text, nothing else. No explanations, no preamble.
"""

def clean_file(raw_path: Path) -> str:
    raw_text = raw_path.read_text(encoding="utf-8")
    
    response = client.models.generate_content(
        model=MODEL,
        contents=f"{SYSTEM_PROMPT}\n\n--- RAW TEXT ---\n{raw_text}",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),  # LOW is faster/cheaper for cleaning
        ),
    )
    return response.text

def run_pipeline():
    raw_files = list(RAW_DIR.glob("*.txt"))
    print(f"Found {len(raw_files)} files to process.")
    
    for raw_file in raw_files:
        output_file = CLEANED_DIR / raw_file.name
        
        if output_file.exists():
            print(f"  Skipping (already cleaned): {raw_file.name}")
            continue
        
        print(f"  Cleaning: {raw_file.name} ...")
        try:
            cleaned = clean_file(raw_file)
            output_file.write_text(cleaned, encoding="utf-8")
            print(f"  Done -> {output_file}")
        except Exception as e:
            print(f"  ERROR on {raw_file.name}: {e}")

if __name__ == "__main__":
    run_pipeline()
    
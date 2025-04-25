import os
import time
import math
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load your OpenRouter API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Set up OpenAI client for OpenRouter
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# ✨ Configuration
TXT_FILE = "en.txt"  # Ensure this file contains one sentence per line
CHUNK_SIZE = 1000
OUTPUT_DIR = "translated_chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGES = [
    "Mandarin Chinese",
    "Spanish",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Hindi",
    "Urdu",
    "Portuguese",
    "Arabic"
]

# 🔁 Safe translation with retry
def translate_text(text, language, max_retries=3):
    prompt = f"Translate the following English text to {language}:\n\n{text}"
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            if response and hasattr(response, "choices") and response.choices:
                return response.choices[0].message.content.strip()
            else:
                raise ValueError("Empty or malformed API response")
        except Exception as e:
            print(f"❌ Error ({language}) on attempt {attempt+1}: {e}")
            time.sleep(2)
    return ""  # Fallback if all retries fail

# 🧾 Load the txt file
with open(TXT_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"📄 Loaded {len(lines)} rows. Splitting into chunks of {CHUNK_SIZE}...")

# 🔁 Process chunks
num_chunks = math.ceil(len(lines) / CHUNK_SIZE)
for lang in LANGUAGES:
    for chunk_id in range(num_chunks):
        chunk_start = chunk_id * CHUNK_SIZE
        chunk_end = min((chunk_id + 1) * CHUNK_SIZE, len(lines))
        chunk_lines = lines[chunk_start:chunk_end]

        print(f"⏳ {lang} - Chunk {chunk_id}: Translating {len(chunk_lines)} lines...")
        translations = []
        for line in tqdm(chunk_lines, desc=f"{lang} - Chunk {chunk_id}"):
            translated = translate_text(line, lang)
            translations.append((line, translated))

        # 📁 Save chunk result
        df_chunk = pd.DataFrame(translations, columns=["English", lang])
        output_path = os.path.join(OUTPUT_DIR, f"{lang.replace(' ', '_')}_chunk_{chunk_id:03}.csv")
        df_chunk.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved: {output_path}")

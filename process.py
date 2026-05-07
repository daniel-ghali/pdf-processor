import requests
import fitz
import os
import uuid
import re

from google import genai
from supabase import create_client, Client

print("🚀 Script started (Gemini Mode with Data Cleaning)")

# Environment Variables
google_api_key = os.environ.get("GOOGLE_API_KEY")
file_url = os.environ.get("FILE_URL")

if file_url:
    file_url = file_url.strip().lstrip("=")

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not all([google_api_key, file_url, supabase_url, supabase_key]):
    print("❌ Missing environment variables")
    exit(1)

# Clients
client = genai.Client(api_key=google_api_key)
supabase: Client = create_client(supabase_url, supabase_key)

# Unique file ID
file_id = str(uuid.uuid4())

print("📄 File URL:", file_url)

# ================================================================
# DATA CLEANING FUNCTIONS
# ================================================================

def is_valid_text(text):
    """Check if text contains bad patterns that should be excluded"""
    bad_patterns = [
        "Sign in",
        "Google Drive",
        "Email or phone",
        "Next",
        "Create account",
        "[image]",
        "Forgot password?",
        "Forgot email?",
        "Not your computer?",
        "Use a private browsing window to sign in.",
        "Learn more",
        "Create account",
        "Sign in",
        "Privacy",
        "Terms"
    ]
    return not any(pattern in text for pattern in bad_patterns)

def clean_content(raw_text):
    """Clean raw text to remove UI noise and improve quality"""
    content = raw_text
    
    # Remove UI noise patterns
    content = re.sub(r'\[image\]', '', content, flags=re.IGNORECASE)
    
    # Remove Google Drive/Sign in sections
    content = re.sub(r'Sign in[\s\S]*?Next', '', content, flags=re.IGNORECASE)
    content = re.sub(r'Google Drive[\s\S]*', '', content, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    content = re.sub(r'\s+', ' ', content)
    content = content.strip()
    
    return content

try:
    # Download PDF
    response = requests.get(file_url)
    response.raise_for_status()

    pdf_bytes = response.content

    print("📦 PDF downloaded:", len(pdf_bytes))

    # Open PDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    print("📑 Pages:", len(doc))

    total_chunks_saved = 0

    for i, page in enumerate(doc):

        text = page.get_text()

        print(f"📝 Page {i} raw length:", len(text))

        # Clean the entire page first
        cleaned_page_text = clean_content(text)
        
        if not is_valid_text(cleaned_page_text):
            print(f"⚠️ Skipping page {i} - contains invalid content")
            continue

        # Chunking with overlap (improved)
        chunk_size = 1000
        chunk_overlap = 100
        
        for j in range(0, len(cleaned_page_text), chunk_size - chunk_overlap):

            chunk = cleaned_page_text[j:j + chunk_size]
            chunk_idx = total_chunks_saved

            # Validate chunk before embedding
            if len(chunk) < 50:
                print(f"⚠️ Skipping chunk {chunk_idx} - too short (<50 chars)")
                continue

            if not is_valid_text(chunk):
                print(f"⚠️ Skipping chunk {chunk_idx} - invalid content")
                continue

            print(f"➡️ Processing chunk {chunk_idx} (length: {len(chunk)})")

            # Generate Embedding
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=chunk,
                config={
                    "output_dimensionality": 768
                }
            )

            embedding = result.embeddings[0].values

            print("🧬 Embedding size:", len(embedding))

            # Save to Supabase
            data = {
                "content": chunk,
                "embedding": embedding,
                "file_id": file_id,
                "source": file_url,
                "chunk_index": chunk_idx
            }

            supabase.table("documents").insert(data).execute()

            print(f"✅ Chunk {chunk_idx} saved successfully")
            total_chunks_saved += 1

    print(f"🎉 Processing complete! Total chunks saved: {total_chunks_saved}")

except Exception as e:
    print("❌ Error:", str(e))
    exit(1)

print("🎉 Finished!")

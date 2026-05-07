import requests
import fitz
import os
from google import genai
from google.genai import types
from supabase import create_client, Client

print("🚀 Script started (Gemini Mode)")

# 1. Environment Validation
google_api_key = os.environ.get("GOOGLE_API_KEY")
file_url = os.environ.get("FILE_URL")
if file_url:
    file_url = file_url.strip().lstrip("=") # Fixes the "=https://" error

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not all([google_api_key, file_url, supabase_url, supabase_key]):
    print("❌ Error: Missing required environment variables")
    exit(1)

# 2. Initialize Clients
client = genai.Client(api_key=google_api_key)
supabase: Client = create_client(supabase_url, supabase_key)
print("📄 File URL:", file_url)

# 3. Download and Process PDF
try:
    response = requests.get(file_url)
    response.raise_for_status()
    pdf_bytes = response.content
    print("📦 PDF downloaded, size:", len(pdf_bytes))

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    print("📑 Pages:", len(doc))

    for i, page in enumerate(doc):
        text = page.get_text()
        print(f"📝 Page {i} text length:", len(text))

        for j in range(0, len(text), 1000):
            chunk = text[j:j+1000]
            chunk_idx = j // 1000
            print(f"➡️ Processing chunk {chunk_idx} (size: {len(chunk)} characters)...")

            # 4. Generate Gemini Embedding (FREE)
            result = client.models.embed_content(
                model="text-embedding-004",
                contents=chunk,
                config=types.EmbedContentConfig(task_type=types.TaskType.RETRIEVAL_DOCUMENT)
            )
            
            if not result.embeddings or len(result.embeddings) == 0:
                print(f"⚠️ Warning: No embeddings returned for chunk {chunk_idx}")
                continue

            embedding = result.embeddings[0].values
            print(f"🧬 Embedding generated (dimensions: {len(embedding)})")

            # 5. Insert into Supabase
            data = {
                "content": chunk,
                "metadata": {"page": i, "source": file_url},
                "embedding": embedding
            }
            
            print(f"📤 Saving chunk {chunk_idx} to Supabase...")
            supabase.table("donbosco_documents").insert(data).execute()
            print(f"✅ Chunk {chunk_idx} saved successfully")

except Exception as e:
    print(f"❌ Critical Error: {str(e)}")
    exit(1)

print("🎉 All documents processed and indexed with Gemini!")

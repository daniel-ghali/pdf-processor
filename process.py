import requests
import fitz
import os
import uuid

from google import genai
from supabase import create_client, Client

print("🚀 Script started (Gemini Mode)")

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

try:
    # Download PDF
    response = requests.get(file_url)
    response.raise_for_status()

    pdf_bytes = response.content

    print("📦 PDF downloaded:", len(pdf_bytes))

    # Open PDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    print("📑 Pages:", len(doc))

    for i, page in enumerate(doc):

        text = page.get_text()

        print(f"📝 Page {i} length:", len(text))

        # Chunking
        for j in range(0, len(text), 1000):

            chunk = text[j:j + 1000]
            chunk_idx = j // 1000

            print(f"➡️ Chunk {chunk_idx}")

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

            print(f"✅ Chunk {chunk_idx} saved")

except Exception as e:
    print("❌ Error:", str(e))
    exit(1)

print("🎉 Finished!")

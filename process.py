import requests
import fitz
import os
import google.generativeai as genai
from supabase import create_client, Client

print("🚀 Script started (Gemini Mode)")

# 1. Environment Validation
google_api_key = os.environ.get("GOOGLE_API_KEY")
file_url = os.environ.get("FILE_URL")
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not all([google_api_key, file_url, supabase_url, supabase_key]):
    print("❌ Error: Missing required environment variables (GOOGLE_API_KEY, FILE_URL, SUPABASE_URL, or SUPABASE_KEY)")
    exit(1)

# 2. Initialize Clients
genai.configure(api_key=google_api_key)
supabase: Client = create_client(supabase_url, supabase_key)
print("📄 File URL:", file_url)

# 3. Download and Process PDF
try:
    pdf_bytes = requests.get(file_url).content
    print("📦 PDF downloaded, size:", len(pdf_bytes))

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    print("📑 Pages:", len(doc))

    for i, page in enumerate(doc):
        text = page.get_text()
        print(f"📝 Page {i} text length:", len(text))

        # Chunking text (approx 2000 chars for Gemini is fine, but keeping 800 for consistency)
        for j in range(0, len(text), 800):
            chunk = text[j:j+800]
            print(f"➡️ Processing chunk {j//800}...")

            # 4. Generate Gemini Embedding (FREE)
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=chunk,
                task_type="retrieval_document"
            )
            embedding = result['embedding']

            # 5. Insert into Supabase
            data = {
                "content": chunk,
                "metadata": {"page": i, "source": file_url},
                "embedding": embedding
            }
            
            supabase.table("donbosco_documents").insert(data).execute()
            print(f"✅ Chunk {j//800} saved to Supabase")

except Exception as e:
    print(f"❌ Critical Error: {str(e)}")
    exit(1)

print("🎉 All documents processed and indexed with Gemini!")

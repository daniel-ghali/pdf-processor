import requests
import fitz
import os
import google.generativeai as genai
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
genai.configure(api_key=google_api_key)
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
            print(f"➡️ Processing chunk {j//1000}...")

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
            print(f"✅ Chunk {j//1000} saved to Supabase")

except Exception as e:
    print(f"❌ Critical Error: {str(e)}")
    exit(1)

print("🎉 All documents processed and indexed with Gemini!")

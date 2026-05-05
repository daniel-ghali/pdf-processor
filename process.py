import os
import gc
import requests
import fitz
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

file_url = os.environ["FILE_URL"]
file_id = os.environ["FILE_ID"]

supabase_url = os.environ["SUPABASE_URL"]
supabase_key = os.environ["SUPABASE_KEY"]

headers = {
    "apikey": supabase_key,
    "Authorization": "Bearer " + supabase_key,
    "Content-Type": "application/json"
}

# --- DEDUP CHECK ---
check = requests.get(
    supabase_url + f"/rest/v1/processed_files?file_id=eq.{file_id}&select=file_id",
    headers=headers,
    timeout=15
)
if check.json():
    print(f"File {file_id} already processed. Skipping.")
    exit(0)

try:
    gc.collect()

    pdf_bytes = requests.get(file_url, timeout=60).content
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    chunks = []

    for page in doc:
        text = page.get_text()
        if not text or len(text.strip()) < 5:
            continue
        for i in range(0, len(text), 800):
            chunk = text[i:i+800].strip()
            if chunk:
                chunks.append(chunk)

    del doc
    gc.collect()

    for chunk in chunks:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        requests.post(
            supabase_url + "/rest/v1/documents",
            headers=headers,
            json={
                "content": chunk,
                "embedding": emb.data[0].embedding,
                "file_id": file_id
            },
            timeout=30
        )

    # --- MARK AS PROCESSED ---
    requests.post(
        supabase_url + "/rest/v1/processed_files",
        headers=headers,
        json={"file_id": file_id},
        timeout=15
    )

    print(f"Done. {len(chunks)} chunks processed.")

except Exception as e:
    print("ERROR:", str(e))
    raise

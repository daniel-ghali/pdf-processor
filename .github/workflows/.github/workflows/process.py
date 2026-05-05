import requests
import fitz
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

file_url = os.environ["FILE_URL"]

pdf_bytes = requests.get(file_url).content

doc = fitz.open(stream=pdf_bytes, filetype="pdf")

chunks = []

# STEP 1: extract text
for page in doc:
    text = page.get_text()

    # chunking
    for i in range(0, len(text), 800):
        chunks.append(text[i:i+800])

# STEP 2: embeddings + send to Supabase
for chunk in chunks:
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )

    requests.post(
        os.environ["SUPABASE_URL"] + "/rest/v1/documents",
        headers={
            "apikey": os.environ["SUPABASE_KEY"],
            "Authorization": "Bearer " + os.environ["SUPABASE_KEY"],
            "Content-Type": "application/json"
        },
        json={
            "content": chunk,
            "embedding": emb.data[0].embedding
        }
    )

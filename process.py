import requests
import fitz
import os
from openai import OpenAI

print("🚀 Script started")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

file_url = os.environ["FILE_URL"]
print("📄 File URL:", file_url)

pdf_bytes = requests.get(file_url).content
print("📦 PDF downloaded, size:", len(pdf_bytes))

doc = fitz.open(stream=pdf_bytes, filetype="pdf")
print("📑 Pages:", len(doc))

for i, page in enumerate(doc):
    text = page.get_text()
    print(f"📝 Page {i} text length:", len(text))

    for j in range(0, len(text), 800):
        chunk = text[j:j+800]

        print("➡️ Sending chunk...")

        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )

        print("✅ Embedding done")

print("🎉 Done")

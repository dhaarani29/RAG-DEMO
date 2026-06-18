import chromadb
from sentence_transformers import SentenceTransformer
import os

client = chromadb.Client()

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="company_docs"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

folder = "data"

for file in os.listdir(folder):

    path = os.path.join(folder, file)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    embedding = model.encode(text).tolist()

    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[file]
    )

print("Documents stored!")
print(collection.count())
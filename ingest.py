import chromadb
from sentence_transformers import SentenceTransformer
import os

# Initialize the ChromaDB client and load the collection
client = chromadb.PersistentClient(path="./chroma_db")
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
folder = "data"

# Check if the collection exists, if not, create it
collection = client.get_or_create_collection(
    name="company_docs"
)

# Ingest documents from the specified folder
for file in os.listdir(folder):

    # Only process text files
    path = os.path.join(folder, file)

    # Read the content of the file
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Generate the embedding for the document
    embedding = model.encode(text).tolist()

    # Store the document and its embedding in the collection
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[file]
    )

# Print the number of documents stored
print("Documents stored!")
print(collection.count())
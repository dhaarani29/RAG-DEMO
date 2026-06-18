import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

client =  chromadb.PersistentClient(path="./chroma_db")

collection_name = "company_docs"
try:
    collection = client.get_collection(collection_name)
except NotFoundError:
    collection = client.create_collection(name=collection_name)
    print(f"Created collection '{collection_name}'")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Document Count:", collection.count())

while True:
    question = input("Ask: ").strip()
    if question.lower() in ("exit", "quit"):
        break

    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    print(results["distances"])

    docs = results.get("documents", [[]])[0]

    if not docs:
        print("\nNo documents retrieved.\n")
        continue

    print("\nRetrieved Documents:\n")
    for doc in docs:
        print(doc)
        print("-" * 40)
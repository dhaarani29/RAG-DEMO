import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer
from ollama import chat

# Initialize the ChromaDB client and load the collection
client =  chromadb.PersistentClient(path="./chroma_db")
collection_name = "company_docs"
model = SentenceTransformer("all-MiniLM-L6-v2")
withLLM = True  # Set to True if you want to use LLM for answering questions

# Check if the collection exists, if not, create it
try:
    collection = client.get_collection(collection_name)
except NotFoundError:
    collection = client.create_collection(name=collection_name)
    print(f"Created collection '{collection_name}'")

# Start the query loop
while True:
    # Get user input
    question = input("Ask: ").strip()
    
    # Exit the loop if the user types "exit" or "quit"
    if question.lower() in ("exit", "quit"):
        break

    # Generate the embedding for the user's question
    query_embedding = model.encode(question).tolist()

    # Query the collection for the most relevant document
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    # withLLM is True, use the retrieved document to answer the question using LLM
    if withLLM:
        context = "\n".join(results["documents"][0])

        prompt = f"""
               You are a company assistant.

                Answer ONLY from the provided context.

                If the answer is not present, say:

                "I couldn't find that information in the provided documents."

                Context:
                {context}

                Question:
                {question}
                """
    
        response = chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        print(response["message"]["content"])

    # If withLLM is False, just display the retrieved documents
    elif not withLLM:
        docs = results.get("documents", [[]])[0]

        if not docs:
            print("\nNo documents retrieved.\n")
            continue

        print("\nRetrieved Documents:\n")
        
        for doc in docs:
            print(doc)
            print("-" * 40)
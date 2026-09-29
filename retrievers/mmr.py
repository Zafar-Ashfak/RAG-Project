from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_PATH = "local-db"

# Step 1: Create documents
docs = [
    Document(metadata={"source": "ml_basics.txt"},
             page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(metadata={"source": "ml_basics.txt"},
             page_content="Gradient descent minimizes the loss function."),
    Document(metadata={"source": "ml_basics.txt"},
             page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(metadata={"source": "neural_networks.txt"},
             page_content="Neural networks use gradient descent for training."),
    Document(metadata={"source": "ml_algorithms.txt"},
             page_content="Support Vector Machines are supervised learning algorithms."),
]

# Step 2: Generate embeddings for each chunk
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 3: Store the embeddings and their corresponding chunks in a vector store
vector_stores = Chroma.from_documents(
    embedding=embedding_model,
    documents=docs,
    persist_directory=DB_PATH
)

# Step 4: Retrieve the most similar chunks and display using similarity search and mmr

print("\n===== Similarity Search Results =====\n")

similarity_retriever = vector_stores.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

similarity_docs = similarity_retriever.invoke("What is gradient descent?")

for doc in similarity_docs:
    print(doc.page_content)

print("\n===== MMR (Maximum Marginal Relevance) Search Results =====\n")

mmr_retriever = vector_stores.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 5
    }
)

mmr_docs = mmr_retriever.invoke("What is gradient descent?")

for doc in mmr_docs:
    print(doc.page_content)

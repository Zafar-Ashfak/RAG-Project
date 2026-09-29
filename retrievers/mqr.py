from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.retrievers import MultiQueryRetriever


FILE_PATH = "../GRU.pdf"
DB_PATH = "mqr_local-db"

# Step 1: Load documents from the source (PDF)
loader = PyPDFLoader(file_path=FILE_PATH)
docs = loader.load()

# Step 2: Split documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20
)

chunks = splitter.split_documents(docs)

# Step 3: Generate embeddings for each chunk
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Store the embeddings and corresponding chunks in the vector store
vector_stores = Chroma.from_documents(
    embedding=embedding_model,
    documents=chunks,
    persist_directory=DB_PATH,
)

retriever = vector_stores.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 5
    }
)

# Step 5: Create a Multi-Query Retriever to generate multiple variations of the user's query
def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        temperature=0
    )

    return ChatHuggingFace(llm=llm)

# Step 6: Retrieve relevant chunks using the generated queries
chat_model = get_llm()
multi_query_retriever =  MultiQueryRetriever.from_llm(
    retriever= retriever,
    llm=chat_model
)

# Step 7: Combine and deduplicate the retrieved documents
query = "What is a recurrent neural network (RNN)? "

retrieved_docs = multi_query_retriever.invoke(query)

# Step 8: Display the retrieved chunks
for i, doc in enumerate(retrieved_docs, start=1):
    print(f"\n{'-' * 20} Document {i} {'-' * 20}")
    print("Metadata:", doc.metadata)
    print(doc.page_content)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_PATH = "deeplearning.pdf"
DB_PATH = "chroma-db"

# Step: 1 - Load PDF
loader = PyPDFLoader(file_path=PDF_PATH)
docs = loader.load()

# Step: 2 - Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Create chunks
chunks = splitter.split_documents(docs)

if not chunks:
    raise ValueError("No text found in the PDF.")

# Step: 3 - Create embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step: 4 - Store in chroma database
vector_stores =Chroma.from_documents(
    embedding=embedding_model,
    documents=chunks,
    persist_directory=DB_PATH,
    collection_name="deep_learning"
)








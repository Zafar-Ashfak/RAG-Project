from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the PDF document and convert it into LangChain Document objects
loader = PyPDFLoader(file_path="../GRU.pdf")

docs = loader.load()

# Split the loaded document into smaller, overlapping text chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

# Iterate through the chunks and display each chunk's content
for index, chunk in enumerate(chunks, start=1):
    print(f"\n----------------- Chunk {index} -----------------")
    print(f"Metadata: {chunk.metadata}")
    print(f"{chunk.page_content}\n")





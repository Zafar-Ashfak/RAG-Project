from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# splitting texts into chunks
splitter = CharacterTextSplitter(
    separator="",
    chunk_size = 10,
    chunk_overlap=2
)

# Load the text file
loader = TextLoader(
    file_path="intro.txt",
    encoding="utf-8"
)

docs = loader.load()
chunks = splitter.split_documents(docs)

for chunk in chunks:
    print(chunk.page_content)
    print("\n\n")
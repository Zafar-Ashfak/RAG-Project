from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Load the text document and convert it into LangChain Document objects
loader = TextLoader(
    file_path="../intro.txt",
    encoding="utf-8"
)

docs = loader.load()

# Split the loaded document into smaller, overlapping text chunks
splitter = CharacterTextSplitter(
    separator="",
    chunk_size = 20,
    chunk_overlap=3
)

chunks = splitter.split_documents(docs)

# Iterate through the chunks and display their content
for chunk in chunks:
    print(chunk.page_content)
    print("\n\n")

from langchain_community.document_loaders import TextLoader

# Load the text file
loader = TextLoader(
    file_path="../genai.txt",
    encoding="utf-8"
)

docs = loader.load()

print(f"\nLength of the document is: {len(docs)}\n\n")
print(docs[0].page_content)
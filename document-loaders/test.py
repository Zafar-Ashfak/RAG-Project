from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    file_path="genai.txt",
    encoding="utf-8"
)

docs = loader.load()

print(docs[0].page_content)
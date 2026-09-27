from langchain_community.document_loaders import WebBaseLoader

url = "https://docs.langchain.com/oss/python/deepagents/rag"

data = WebBaseLoader(url)

docs = data.load()

print(docs)
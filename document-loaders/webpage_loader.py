from langchain_community.document_loaders import WebBaseLoader

url = "https://docs.langchain.com/oss/python/deepagents/rag"

# Loading webpage (Retrieval Augmented Generation (RAG) with Deep Agents)
data = WebBaseLoader(url)

docs = data.load()

print(f"\nLength of the document is: {len(docs)}\n\n")
print(docs[0].page_content)
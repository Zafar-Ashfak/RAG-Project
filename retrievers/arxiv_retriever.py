import arxiv
from langchain_community.retrievers import ArxivRetriever

# Configure the arXiv API to use HTTPS
arxiv.Client.query_url_format = "https://export.arxiv.org/api/query?{}"

retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True
)

# Search for research papers related to Large Language Models
retrieved_docs = retriever.invoke("Large Language Model")

# Display the retrieved documents
for i, doc in enumerate(retrieved_docs):
    print(f"\n{'-' * 40} Document {i + 1} {'-' * 40}")
    print(f"Title: {doc.metadata.get('Title')}")
    print(f"Authors: {doc.metadata.get('Authors')}")
    print(doc.page_content)
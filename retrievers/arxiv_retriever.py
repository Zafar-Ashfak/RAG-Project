import arxiv
from langchain_community.retrievers import ArxivRetriever

arxiv.Client.query_url_format = "https://export.arxiv.org/api/query?{}"

retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True
)

docs = retriever.invoke("Large Language Model")

for doc in docs:
    print(doc.metadata)
    print(doc.page_content)
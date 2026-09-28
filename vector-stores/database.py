from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

docs = [
        Document(
            metadata={"source": "AI_book"},
            page_content="Python is widely used in Artificial Intelligence."),
        Document(
            metadata={"source": "DataScience_book"},
            page_content="Pandas is used for data analysis in Python."),
        Document(
            metadata={"source": "DL_book"},
            page_content="Neural networks are used in deep learning.")
]

embedding = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",

)

document_vectors = embedding.embed_documents(doc.page_content for doc in docs)

print(document_vectors)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

docs = [
    Document(
        metadata={"source": "AI_book"},
        page_content="Python is widely used in Artificial Intelligence."
    ),

    Document(
        metadata={"source": "DataScience_book"},
        page_content="Pandas is used for data analysis in Python."
    ),

    Document(
        metadata={"source": "DL_book"},
        page_content="Neural networks are used in deep learning."
    )
]

embedding_model = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstores = Chroma(
    documents = docs,
    embeddings=embedding_model,
    persist_directory="chroma-db"
)


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

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
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstores = Chroma.from_documents(
    embedding=embedding_model,
    documents=docs,
    persist_directory="../chroma-db"
)

results = vectorstores.similarity_search("Which language is used in Artificial Intelligence?", k=2)

for result in results:
    print(result.page_content)
    print(result.metadata)

retriever = vectorstores.as_retriever()

similar_chunks = retriever.invoke("Explain Deep Learning.")

for chunk in similar_chunks:
    print(chunk.page_content)


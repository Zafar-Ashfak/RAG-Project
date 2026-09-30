from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from create_database import embedding_model

vector_stores = Chroma(
    persist_directory="chroma-db",
    collection_name="deep_learning",
    embedding_function=embedding_model
)

retriever = vector_stores.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        temperature=0.2
    )

    return ChatHuggingFace(llm=llm)


prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant specializing in deep learning.

        Instructions:
        1. Answer the user's question using ONLY the provided context.
        2. Explain the answer clearly and in detail.
        3. Organize the response using headings, bullet points,
           or examples when appropriate.
        4. If the context contains only partial information,
           explain what can be answered from the context.
        5. If the answer cannot be found in the context,
           respond with:
           "I could not find the answer in the document."
        6. Do not make up facts or use external knowledge.
        """
    ),
    (
        "human",
        """
        Context:
        {context}

        Question:
        {query}
        """
    )
])

def main():
    print("What's in your mind")
    print("Enter quit, exit or 0 to close the chat")
    chat_model = get_llm()

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["quit", "exit", "0"]:
            break

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt_template.invoke({
            "context": context,
            "query": query
        })

        response = chat_model.invoke(final_prompt)

        print(f"\n AI: {response.content}")


main()

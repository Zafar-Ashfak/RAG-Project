from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate

def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        temperature=0
    )

    return ChatHuggingFace(llm=llm)

chat_prompt =  ChatPromptTemplate.from_messages([
    ("system", """
        You are a helpful AI assistant.
        Summarize the entire text clearly and concisely.
    """),
    ("human", "{docs}")
])

# Loading text file
# data = TextLoader(file_path="document-loaders/genai.txt", encoding="utf-8")

# loading pdf file
data = PyPDFLoader(file_path="document-loaders/GRU.pdf")
docs = data.load()

def main():
    model = get_llm()

    final_prompt = chat_prompt.invoke({
        # "docs" : docs[0].page_content # Sending only the first page
        "docs" : docs # Sending the whole page to check if we get any context window error
    })

    response = model.invoke(final_prompt)
    print(response)

    for line in response.content.splitlines():
        print(line)

main()
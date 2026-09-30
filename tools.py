from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

FILE_PATH = "deeplearning.pdf"

# Step 1: Load documents from the source (PDF)
loader = PyPDFLoader(file_path=FILE_PATH)
docs = loader.load()

# Step 2: Split documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# Creating a chat model
def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        temperature=0.2,
        max_new_tokens=1500
    )

    return ChatHuggingFace(llm=llm)

# Creating prompt template
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant specializing in understanding
        and summarizing documents.

        Your task is to analyze the provided document and create
        a clear, accurate, and well-structured summary.

        Follow these instructions:
        1. Identify the main topic and purpose of the document.
        2. Explain the key concepts and important points.
        3. Preserve important technical terms, definitions, and formulas.
        4. Organize the summary using appropriate headings and bullet points.
        5. Explain complex concepts in simple, beginner-friendly language.
        6. Do not repeat information unnecessarily.
        7. Do not add facts or information that are not present in the document.
        8. If something is unclear or missing, state that explicitly.

        Make the summary comprehensive but concise.
        """
    ),
    (
        "human",
        """
        Please summarize the following document:

        {docs}
        """
    )
])


def main():

    model = get_llm()

    final_prompt = prompt_template.invoke({
        "docs": docs
    })

    response = model.invoke(final_prompt)
    print(response.content)

main()
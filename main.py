from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from dotenv import load_dotenv
load_dotenv()

def get_llm():
    return HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        temperature=0
    )

def main():
    prompt = input("What's in your mind!\n")
    llm = get_llm()
    model = ChatHuggingFace(llm=llm)
    response = model.invoke(prompt)
    print(response)

main()
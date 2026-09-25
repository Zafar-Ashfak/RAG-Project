from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

def get_llm():
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )


def main():
    prompt = input("What's in your mind!\n")
    llm = get_llm()
    response = llm.invoke(prompt)
    print(response)

main()
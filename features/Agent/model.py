from langchain_ollama import ChatOllama

model = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.6,
    reasoning=True
)
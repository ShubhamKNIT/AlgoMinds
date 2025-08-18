from langchain_ollama import OllamaEmbeddings

embedding_model = OllamaEmbeddings(
    model="mxbai-embed-large:latest",
    temperature=0.6,
    top_k=3
)

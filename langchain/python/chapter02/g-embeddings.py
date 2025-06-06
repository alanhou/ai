from langchain_openai import OpenAIEmbeddings

model = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = model.embed_documents([
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"
])

print(embeddings)
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./test.txt")
docs = loader.load()

print(docs)
"""
1. Ensure docker is installed and running (https://docs.docker.com/get-docker/)
2. pip install -qU langchain_postgres
3. Run the following command to start the postgres container:
   
docker run \
    --name pgvector-container \
    -e POSTGRES_USER=langchain \
    -e POSTGRES_PASSWORD=langchain \
    -e POSTGRES_DB=langchain \
    -p 6024:5432 \
    -d pgvector/pgvector:pg16
4. Use the connection string below for the postgres container

"""

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain


# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

# Load the document, split it into chunks
raw_documents = TextLoader('./test.txt', encoding='utf-8').load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# Create embeddings for the documents
embeddings_model = OpenAIEmbeddings()

db = PGVector.from_documents(
    documents, embeddings_model, connection=connection)

# create retriever to retrieve 2 relevant documents
retriever = db.as_retriever(search_kwargs={"k": 2})

query = 'Who are the key figures in the ancient greek history of philosophy?'

# fetch relevant documents
docs = retriever.invoke(query)

print(docs[0].page_content)

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
llm_chain = prompt | llm

# answer the question based on relevant documents
result = llm_chain.invoke({"context": docs, "question": query})

print(result)
print("\n\n")

# Run again but this time encapsulate the logic for efficiency

# @chain decorator transforms this function into a LangChain runnable,
# making it compatible with LangChain's chain operations and pipeline

print("Running again but this time encapsulate the logic for efficiency\n")


@chain
def qa(input):
    # fetch relevant documents
    docs = retriever.invoke(input)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer


# run it
result = qa.invoke(query)
print(result.content)
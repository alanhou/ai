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

# Query starts with irrelevant information before asking the relevant question
query = 'Today I woke up and brushed my teeth, then I sat down to read the news. But then I forgot the food on the cooker. Who are some key figures in the ancient greek history of philosophy?'

# fetch relevant documents
docs = retriever.invoke(query)

print(docs[0].page_content)
print("\n\n")

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# Run again but this time encapsulate the logic for efficiency

# @chain decorator transforms this function into a LangChain runnable,
# making it compatible with LangChain's chain operations and pipeline

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

print("\nRewrite the query to improve accuracy\n")

rewrite_prompt = ChatPromptTemplate.from_template(
    """Provide a better search query for web search engine to answer the given question, end the queries with ’**’. Question: {x} Answer:""")


def parse_rewriter_output(message):
    return message.content.strip('"').strip("**")


rewriter = rewrite_prompt | llm | parse_rewriter_output


@chain
def qa_rrr(input):
    # rewrite the query
    new_query = rewriter.invoke(input)
    print("Rewritten query: ", new_query)
    # fetch relevant documents
    docs = retriever.invoke(new_query)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer


print("\nCall model again with rewritten query\n")

# call model again with rewritten query
result = qa_rrr.invoke(query)
print(result.content)
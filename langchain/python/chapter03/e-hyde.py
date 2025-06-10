from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser

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
retriever = db.as_retriever(search_kwargs={"k": 5})

prompt_hyde = ChatPromptTemplate.from_template(
    """Please write a passage to answer the question.\n Question: {question} \n Passage:""")

generate_doc = (prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser())

"""
Next, we take the hypothetical document generated above and use it as input to the retriever, 
which will generate its embedding and search for similar documents in the vector store:
"""
retrieval_chain = generate_doc | retriever

query = "Who are some lesser known philosophers in the ancient greek history of philosophy?"

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


@chain
def qa(input):
    # fetch relevant documents from the hyde retrieval chain defined earlier
    docs = retrieval_chain.invoke(input)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer


print("Running hyde\n")
result = qa.invoke(query)
print("\n\n")
print(result.content)
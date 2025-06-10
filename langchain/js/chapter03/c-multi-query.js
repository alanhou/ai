/** 
1. Ensure docker is installed and running (https://docs.docker.com/get-docker/)
2. Run the following command to start the postgres container:
   
docker run \
    --name pgvector-container \
    -e POSTGRES_USER=langchain \
    -e POSTGRES_PASSWORD=langchain \
    -e POSTGRES_DB=langchain \
    -p 6024:5432 \
    -d pgvector/pgvector:pg16
3. Use the connection string below for the postgres container
*/

import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { OpenAIEmbeddings } from '@langchain/openai';
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI } from '@langchain/openai';
import { RunnableLambda } from '@langchain/core/runnables';

const connectionString =
  'postgresql://langchain:langchain@localhost:6024/langchain';
// Load the document, split it into chunks
const loader = new TextLoader('./test.txt');
const raw_docs = await loader.load();
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const splitDocs = await splitter.splitDocuments(raw_docs);

// embed each chunk and insert it into the vector store
const model = new OpenAIEmbeddings();

const db = await PGVectorStore.fromDocuments(splitDocs, model, {
  postgresConnectionOptions: {
    connectionString,
  },
});

// retrieve 2 relevant documents from the vector store
const retriever = db.asRetriever({ k: 2 });
/**
 * Provide retrieved docs as context to the LLM to answer a user's question
 */
const llm = new ChatOpenAI({ temperature: 0, modelName: 'gpt-4o-mini' });

const perspectivesPrompt = ChatPromptTemplate.fromTemplate(
  `You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {question}`
);

const queryGen = perspectivesPrompt.pipe(llm).pipe((message) => {
  return message.content.split('\n');
});

/**
 * This chain retrieves and combines the documents from the vector store for each query
 */
const retrievalChain = queryGen
  .pipe(retriever.batch.bind(retriever))
  .pipe((documentLists) => {
    const dedupedDocs = {};
    documentLists.flat().forEach((doc) => {
      dedupedDocs[doc.pageContent] = doc;
    });
    return Object.values(dedupedDocs);
  });

const prompt = ChatPromptTemplate.fromTemplate(
  'Answer the question based only on the following context:\n {context}\n\nQuestion: {question}'
);

console.log('Running multi query qa\n');
const multiQueryQa = RunnableLambda.from(async (input) => {
  // fetch relevant documents
  const docs = await retrievalChain.invoke({ question: input });
  // format prompt
  const formatted = await prompt.invoke({ context: docs, question: input });
  // generate answer
  const answer = await llm.invoke(formatted);
  return answer;
});

const result = await multiQueryQa.invoke(
  'Who are the key figures in the ancient greek history of philosophy?'
);

console.log(result);
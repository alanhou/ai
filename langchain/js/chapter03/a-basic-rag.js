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

const query =
  'Who are the key figures in the ancient greek history of philosophy?';

// fetch relevant documents
const docs = await retriever.invoke(query);

console.log(
  `fetched document based on similarity search query:\n ${docs[0].pageContent}\n\n`
);

/**
 * Provide retrieved docs as context to the LLM to answer a user's question
 */
const prompt = ChatPromptTemplate.fromTemplate(
  'Answer the question based only on the following context:\n {context}\n\nQuestion: {question}'
);

const llm = new ChatOpenAI({ temperature: 0, modelName: 'gpt-3.5-turbo' });
const chain = prompt.pipe(llm);

const result = await chain.invoke({
  context: docs,
  question: query,
});

console.log(result);
console.log('\n\n');

// run again but this time encapsulate the logic for efficiency

console.log(
  'Running again but this time encapsulate the logic for efficiency\n'
);
const qa = RunnableLambda.from(async (input) => {
  // fetch relevant documents
  const docs = await retriever.invoke(input);
  // format prompt
  const formatted = await prompt.invoke({ context: docs, question: input });
  // generate answer
  const answer = await llm.invoke(formatted);
  return answer;
});

const finalResult = await qa.invoke(query);
console.log(finalResult.content);
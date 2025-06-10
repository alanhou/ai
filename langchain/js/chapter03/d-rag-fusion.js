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
  `You are a helpful assistant that generates multiple search queries based on a single input query. \n Generate multiple search queries related to: {question} \n Output (4 queries):`
);
const queryGen = perspectivesPrompt.pipe(llm).pipe((message) => {
  return message.content.split('\n');
});

function reciprocalRankFusion(results, k = 60) {
  // Initialize a dictionary to hold fused scores for each document
  // Documents will be keyed by their contents to ensure uniqueness
  const fusedScores = {};
  const documents = {};
  results.forEach((docs) => {
    docs.forEach((doc, rank) => {
      // Use the document contents as the key for uniqueness
      const key = doc.pageContent;
      // If the document hasn't been seen yet,
      // - initialize score to 0
      // - save it for later
      if (!(key in fusedScores)) {
        fusedScores[key] = 0;
        documents[key] = 0;
      }
      // Update the score of the document using the RRF formula:
      // 1 / (rank + k)
      fusedScores[key] += 1 / (rank + k);
    });
  });
  // Sort the documents based on their fused scores in descending order to get the final reranked results
  const sorted = Object.entries(fusedScores).sort((a, b) => b[1] - a[1]);
  // retrieve the corresponding doc for each key
  return sorted.map(([key]) => documents[key]);
}

const prompt = ChatPromptTemplate.fromTemplate(
  'Answer the question based only on the following context:\n {context}\n\nQuestion: {question}'
);

const retrievalChain = queryGen
  .pipe(retriever.batch.bind(retriever))
  .pipe(reciprocalRankFusion);

console.log('Running rag fusion\n');
const ragFusion = RunnableLambda.from(async (input) => {
  // fetch relevant documents
  const docs = await retrievalChain.invoke({ question: input });
  // format prompt
  const formatted = await prompt.invoke({ context: docs, question: input });
  // generate answer
  const answer = await llm.invoke(formatted);
  return answer;
});

const result = await ragFusion.invoke(
  'Who are the key figures in the ancient greek history of philosophy?'
);

console.log(result);
/** 
1. 安装并运行 docker (https://docs.docker.com/get-docker/)
2. 运行如下命令启动 postgres 容器:
   
docker run \
    --name pgvector-container \
    -e POSTGRES_USER=langchain \
    -e POSTGRES_PASSWORD=langchain \
    -e POSTGRES_DB=langchain \
    -p 6024:5432 \
    -d pgvector/pgvector:pg16
3. 使用以下连接字符串连接postgres容器
*/

import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { OpenAIEmbeddings } from '@langchain/openai';
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector';
import { v4 as uuidv4 } from 'uuid';

const connectionString =
  'postgresql://langchain:langchain@localhost:6024/langchain';
// Load the document, split it into chunks
const loader = new TextLoader('./test.txt');
const raw_docs = await loader.load();
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const docs = await splitter.splitDocuments(raw_docs);

// embed each chunk and insert it into the vector store
const model = new OpenAIEmbeddings();
const db = await PGVectorStore.fromDocuments(docs, model, {
  postgresConnectionOptions: {
    connectionString,
  },
});

console.log('Vector store created successfully');

const results = await db.similaritySearch('query', 4);

console.log(`Similarity search results: ${JSON.stringify(results)}`);

console.log('Adding documents to the vector store');

const ids = [uuidv4(), uuidv4()];

await db.addDocuments(
  [
    {
      pageContent: 'there are cats in the pond',
      metadata: { location: 'pond', topic: 'animals' },
    },
    {
      pageContent: 'ducks are also found in the pond',
      metadata: { location: 'pond', topic: 'animals' },
    },
  ],
  { ids }
);

console.log('Documents added successfully');

await db.delete({ ids: [ids[1]] });

console.log('second document deleted successfully');
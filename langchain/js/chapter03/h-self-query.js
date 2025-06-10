import { ChatOpenAI } from '@langchain/openai';
import { SelfQueryRetriever } from 'langchain/retrievers/self_query';
import { FunctionalTranslator } from '@langchain/core/structured_query';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { Document } from 'langchain/document';
import { AttributeInfo } from 'langchain/chains/query_constructor';
import { OpenAIEmbeddings } from '@langchain/openai';
/**
 * First, we create a bunch of documents. You can load your own documents here instead.
 * Each document has a pageContent and a metadata field. Make sure your metadata matches the AttributeInfo below.
 */
const docs = [
  new Document({
    pageContent:
      'A bunch of scientists bring back dinosaurs and mayhem breaks loose',
    metadata: {
      year: 1993,
      rating: 7.7,
      genre: 'science fiction',
      length: 122,
    },
  }),
  new Document({
    pageContent:
      'Leo DiCaprio gets lost in a dream within a dream within a dream within a ...',
    metadata: {
      year: 2010,
      director: 'Christopher Nolan',
      rating: 8.2,
      length: 148,
    },
  }),
  new Document({
    pageContent:
      'A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea',
    metadata: { year: 2006, director: 'Satoshi Kon', rating: 8.6 },
  }),
  new Document({
    pageContent:
      'A bunch of normal-sized women are supremely wholesome and some men pine after them',
    metadata: {
      year: 2019,
      director: 'Greta Gerwig',
      rating: 8.3,
      length: 135,
    },
  }),
  new Document({
    pageContent: 'Toys come alive and have a blast doing so',
    metadata: { year: 1995, genre: 'animated', length: 77 },
  }),
  new Document({
    pageContent: 'Three men walk into the Zone, three men walk out of the Zone',
    metadata: {
      year: 1979,
      director: 'Andrei Tarkovsky',
      genre: 'science fiction',
      rating: 9.9,
    },
  }),
];

const llm = new ChatOpenAI({ modelName: 'gpt-4o-mini', temperature: 0 });

const embeddings = new OpenAIEmbeddings();

const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);

/** * First, we define the attributes we want to be able to query on. * in this case, we want to be able to query on the genre, year, director,     rating, and length of the movie. * We also provide a description of each attribute and the type of the attribute. * This is used to generate the query prompts. */

const fields = [
  {
    name: 'genre',
    description: 'The genre of the movie',
    type: 'string or array of strings',
  },
  {
    name: 'year',
    description: 'The year the movie was released',
    type: 'number',
  },
  {
    name: 'director',
    description: 'The director of the movie',
    type: 'string',
  },
  {
    name: 'rating',
    description: 'The rating of the movie (1-10)',
    type: 'number',
  },
  {
    name: 'length',
    description: 'The length of the movie in minutes',
    type: 'number',
  },
];

const attributeInfos = fields.map(
  (field) => new AttributeInfo(field.name, field.description, field.type)
);
const description = 'Brief summary of a movie';
const selfQueryRetriever = SelfQueryRetriever.fromLLM({
  llm,
  vectorStore,
  description,
  attributeInfo: attributeInfos,
  /**
   * We need to use a translator that translates the queries into a
   * filter format that the vector store can understand. LangChain provides one
   * here.
   */
  structuredQueryTranslator: new FunctionalTranslator(),
});

const result = await selfQueryRetriever.invoke(
  'Which movies are less than 90 minutes?'
);
console.log(result);
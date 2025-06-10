import { cosineSimilarity } from '@langchain/core/utils/math';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableLambda } from '@langchain/core/runnables';

const physicsTemplate = `You are a very smart physics professor. You are great     at answering questions about physics in a concise and easy-to-understand     manner. When you don't know the answer to a question, you admit that you don't know. Here is a question: {query}`;

const mathTemplate = `You are a very good mathematician. You are great at answering     math questions. You are so good because you are able to break down hard     problems into their component parts, answer the component parts, and then     put them together to answer the broader question. Here is a question: {query}`;

const embeddings = new OpenAIEmbeddings();

const promptTemplates = [physicsTemplate, mathTemplate];

const promptEmbeddings = await embeddings.embedDocuments(promptTemplates);

const promptRouter = RunnableLambda.from(async (query) => {
  // Embed question
  const queryEmbedding = await embeddings.embedQuery(query);
  // Compute similarity
  const similarities = cosineSimilarity([queryEmbedding], promptEmbeddings)[0];
  // Pick the prompt most similar to the input question
  const mostSimilar =
    similarities[0] > similarities[1] ? promptTemplates[0] : promptTemplates[1];
  console.log(
    `Using ${mostSimilar === promptTemplates[0] ? 'PHYSICS' : 'MATH'}`
  );
  return PromptTemplate.fromTemplate(mostSimilar).invoke({ query });
});

const semanticRouter = promptRouter.pipe(
  new ChatOpenAI({ modelName: 'gpt-4o-mini', temperature: 0 })
);

const result = await semanticRouter.invoke('What is a black hole');
console.log('\nSemantic router result: ', result);
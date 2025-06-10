import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';
import { ChatPromptTemplate } from '@langchain/core/prompts';

const routeQuery = z
  .object({
    datasource: z
      .enum(['python_docs', 'js_docs'])
      .describe(
        'Given a user question, choose which datasource would be most relevant for answering their question'
      ),
  })
  .describe('Route a user query to the most relevant datasource.');

const llm = new ChatOpenAI({ model: 'gpt-4o-mini', temperature: 0 });
// withStructuredOutput is a method that allows us to use the structured output of the model
const structuredLlm = llm.withStructuredOutput(routeQuery, {
  name: 'RouteQuery',
});

const prompt = ChatPromptTemplate.fromMessages([
  [
    'system',
    `You are an expert at routing a user question to the appropriate data source. Based on the programming language the question is referring to, route it to the relevant data source.`,
  ],
  ['human', '{question}'],
]);

const router = prompt.pipe(structuredLlm);

const question = `Why doesn't the following code work: 
from langchain_core.prompts 
import ChatPromptTemplate 
prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"]) 
prompt.invoke("french") `;

const result = await router.invoke({ question });

console.log('Routing to: ', result);

/** Once we’ve extracted the relevant data source, we can pass the value into another function to execute additional logic as required: */

const chooseRoute = (result) => {
  if (result.datasource.toLowerCase().includes('python_docs')) {
    return 'chain for python_docs';
  } else {
    return 'chain for js_docs';
  }
};

const fullChain = router.pipe(chooseRoute);

const finalResult = await fullChain.invoke({ question });

console.log('Choose route: ', finalResult);
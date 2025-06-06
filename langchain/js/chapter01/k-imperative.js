import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableLambda } from '@langchain/core/runnables';

// the building blocks

const template = ChatPromptTemplate.fromMessages([
  ['system', 'You are a helpful assistant.'],
  ['human', '{question}'],
]);

const model = new ChatOpenAI("gpt-4o-mini");

// combine them in a function
// RunnableLambda adds the same Runnable interface for any function you write

const chatbot = RunnableLambda.from(async (values) => {
  const prompt = await template.invoke(values);
  return await model.invoke(prompt);
});

// use it

const response = await chatbot.invoke({
  question: 'Which model providers offer LLMs?',
});
console.log(response.content);
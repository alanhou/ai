import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';

const model = new ChatOpenAI({ model: 'gpt-4o-mini' });
const prompt = [new HumanMessage('What is the capital of France?')];

const response = await model.invoke(prompt);
console.log(response.content);
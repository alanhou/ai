import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';

const model = new ChatOpenAI({ model: 'gpt-4o-mini' });
const prompt = [
  new SystemMessage(
    'You are a helpful assistant that responds to questions with three exclamation marks.'
  ),
  new HumanMessage('What is the capital of France?'),
];

const response = await model.invoke(prompt);
console.log(response.content);
import { ChatOpenAI } from '@langchain/openai';

const model = new ChatOpenAI({ model: 'gpt-4o-mini' });

const response = await model.invoke('The sky is');
console.log(response.content);
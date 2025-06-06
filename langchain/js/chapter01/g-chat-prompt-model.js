import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI } from '@langchain/openai';

const model = new ChatOpenAI("gpt-4o-mini");
const template = ChatPromptTemplate.fromMessages([
  [
    'system',
    'Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don\'t know".',
  ],
  ['human', 'Context: {context}'],
  ['human', 'Question: {question}'],
]);

const prompt = await template.invoke({
  context:
    "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
  question: 'Which model providers offer LLMs?',
});

const response = await model.invoke(prompt);
console.log(response.content);
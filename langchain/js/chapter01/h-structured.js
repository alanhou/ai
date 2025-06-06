import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';

const answerSchema = z
  .object({
    answer: z.string().describe("The answer to the user's question"),
    justification: z.string().describe('Justification for the answer'),
  })
  .describe(
    "An answer to the user's question along with justification for the answer."
  );

const model = new ChatOpenAI({
  model: 'gpt-4o-mini',
  temperature: 0,
}).withStructuredOutput(answerSchema);

const response = await model.invoke(
  'What weighs more, a pound of bricks or a pound of feathers'
);
console.log(response);
import { ChatOpenAI } from '@langchain/openai';

const model = new ChatOpenAI("gpt-4o-mini")

const response = await model.invoke('Hi there!');
console.log(response.content);
// Hello! How can I assist you today?

const completions = await model.batch(['Hi there!', 'Bye!']);
console.log(completions.map(completion => completion.content));
// [ 'Hello! How can I assist you today?', 'Goodbye! Have a great day!' ]

for await (const token of await model.stream('Bye!')) {
    console.log(token.content);
    // Good
    // bye
    // !
    // Have
    // a
    // great
    // day
    // !
}
import { CommaSeparatedListOutputParser } from '@langchain/core/output_parsers';

const parser = new CommaSeparatedListOutputParser();

const response = await parser.invoke('apple, banana, cherry');
console.log(response);
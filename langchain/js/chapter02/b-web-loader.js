// 安装cheerio: npm install cheerio

import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';

const loader = new CheerioWebBaseLoader('https://www.langchain.com/');
const docs = await loader.load();

console.log(docs);
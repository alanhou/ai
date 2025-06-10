/*
The below example will use a SQLite connection with the Chinook database, which is a sample database that represents a digital media store. Follow these installation steps to create Chinook.db in the same directory as this notebook. You can also download and build the database via the command line:

```bash
curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db

```

Afterwards, place `Chinook.db` in the same directory where this code is running.

*/

import { ChatOpenAI } from '@langchain/openai';
import { createSqlQueryChain } from 'langchain/chains/sql_db';
import { SqlDatabase } from 'langchain/sql_db';
import { DataSource } from 'typeorm';
import { QuerySqlTool } from 'langchain/tools/sql';

const datasource = new DataSource({
  type: 'sqlite',
  database: 'Chinook.db', //this should be the path to the db
});
const db = await SqlDatabase.fromDataSourceParams({
  appDataSource: datasource,
});
//test that the db is working
await db.run('SELECT * FROM Artist LIMIT 10;');

const llm = new ChatOpenAI({ modelName: 'gpt-4o-mini', temperature: 0 });
// convert question to sql query
const writeQuery = await createSqlQueryChain({ llm, db, dialect: 'sqlite' });
// execute query
const executeQuery = new QuerySqlTool(db);
// combined
const chain = writeQuery.pipe(executeQuery);

const result = await chain.invoke({
  question: 'How many employees are there?',
});
console.log(result);
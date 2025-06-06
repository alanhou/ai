from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
markdown_text = """
# LangChain

⚡ Building applications with LLMs through composability ⚡

## Quick Install

```bash
pip install langchain
```

As an open source project in a rapidly developing field, we are extremely open 
    to contributions.
"""

md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
)

md_docs = md_splitter.create_documents(
    [markdown_text], [{"source": "https://www.langchain.com"}])

print(md_docs)
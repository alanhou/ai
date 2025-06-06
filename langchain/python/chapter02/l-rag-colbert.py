"""
- Windows is not supported. RAGatouille doesn't appear to work outside WSL and has issues with WSL1. Some users have had success running RAGatouille in WSL2.
- Only on python.
- Read full docs here: https://github.com/AnswerDotAI/RAGatouille/blob/8183aad64a9a6ba805d4066dcab489d97615d316/README.md

- To install run:

```bash
pip install -U ragatouille transformers
```
"""
from ragatouille import RAGPretrainedModel
import requests

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")


def get_wikipedia_page(title: str):
    """
    Retrieve the full text content of a Wikipedia page.
    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    # Wikipedia API endpoint
    URL = "https://en.wikipedia.org/w/api.php"
    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }
    # Custom User-Agent header to comply with Wikipedia's best practices
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1"}
    response = requests.get(URL, params=params, headers=headers)
    data = response.json()
    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None


full_document = get_wikipedia_page("Hayao_Miyazaki")
# Create an index
RAG.index(
    collection=[full_document],
    index_name="Miyazaki-123",
    max_document_length=180,
    split_documents=True,
)
# query
results = RAG.search(query="What animation studio did Miyazaki found?", k=3)

print(results)

# Alternative: Utilize langchain retriever
retriever = RAG.as_langchain_retriever(k=3)
retriever.invoke("What animation studio did Miyazaki found?")
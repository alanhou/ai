from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

response = parser.invoke("apple, banana, cherry")
print(response)

from langchain_openai import ChatOpenAI 

model = ChatOpenAI(model="gpt-4o-mini") 
response = model.invoke("The sky is")
print(response.content)

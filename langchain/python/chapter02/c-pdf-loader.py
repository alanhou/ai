# 安装 pdf 解析库 
# pip install pypdf
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('./test.pdf')
pages = loader.load()

print(pages)
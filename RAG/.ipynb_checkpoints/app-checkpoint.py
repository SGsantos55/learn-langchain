from dotenv import load_dotenv
from langchain.document_loaders import TextLoader

load_dotenv()
loader = TextLoader("data.txt", encoding="utf-8")
docs = loader.load()

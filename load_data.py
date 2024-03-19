from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import openai
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv() # take env variables from .env file

### START -- Mongo DB Cloud database connection
mongo_url = os.getenv("mongo_url")
print("mongo_url=", mongo_url)
client = MongoClient(mongo_url)
dbName = "rag_qna"
collectionName = "collection_of_text_blobs_with_chunks"
collection = client[dbName][collectionName]
### END -- Mongo DB Cloud database connection

### START -- data load from directory
loader = DirectoryLoader('./data',glob="./*.txt", show_progress=True)
data = loader.load()
print("Length of data :", len(data))
text = data[0].page_content
###print(text)
### END -- data load from directory

### START -- create openai embeddings
openai_api_key = os.getenv("openai_api_key")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
### END -- create openai embeddings

print("#### Semantic Chunking ####")
# Percentile - all differences between sentences are calculated, and then any difference greater than the X percentile is split
text_splitter = SemanticChunker(embeddings)
text_splitter = SemanticChunker(
    embeddings, breakpoint_threshold_type="percentile" # "standard_deviation", "interquartile"
)
documents = text_splitter.create_documents([text])
print("Length of documents [No. of chunks] :", len(documents))
###print(documents)

### START -- create the vector store
vectorStore = MongoDBAtlasVectorSearch.from_documents(documents, embeddings, collection=collection)
### END -- create the vector store

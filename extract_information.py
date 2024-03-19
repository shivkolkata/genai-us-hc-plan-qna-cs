from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
###from langchain.chains import RetrievalQAWithSourcesChain
import gradio as gr
from gradio.themes.base import Base

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

### START -- create openai embeddings
openai_api_key = os.getenv("openai_api_key")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
### END -- create openai embeddings

### START -- create the vector store
vectorStore = MongoDBAtlasVectorSearch(collection, embeddings , index_name="rag_qna_vector_index_chunks")
### END -- create the vector store

def query_data(query):
    ###print(query)
    ### Search the vector store with the query
    docs = vectorStore.similarity_search(query, K=1)
    ### print(docs)
    ### Get the first and the most relevant search result
    as_output = docs[0].page_content
    llm = OpenAI(openai_api_key=openai_api_key,temperature = 0, max_tokens=2048)
    retriever = vectorStore.as_retriever()
    ###chain_type_kwargs = {"prompt": query}

    ###qa = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever = retriever, verbose=True,reduce_k_below_max_tokens=True)

    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever = retriever)
    retriever_output = qa.invoke(query)
    return as_output, retriever_output

with gr.Blocks(theme=gr.themes.Soft(),title="Question Answering App using MongoDB Atlas Vector Search and RAG (Ask about Medicare advantage plan)") as demo:
    gr.Markdown(
        """
        # Question Answering App using MongoDB Atlas Vector Search and RAG (Ask about Medicare advantage plan)
        """)
    textbox = gr.Textbox(label="Enter your question", value="What is Medicare advantage ?")
    with gr.Row():
        button = gr.Button("Submit", variant="Primary")
    with gr.Column():
        output1 = gr.Textbox(lines=1, max_lines=10, label="Raw Vector Search output")   
        output2 = gr.Textbox(lines=1, max_lines=10, label="LLM output")
    button.click(query_data, textbox, outputs=[output1, output2])
demo.launch()


    
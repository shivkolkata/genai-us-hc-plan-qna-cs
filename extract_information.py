from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
#from langchain_mongodb import MongoDBAtlasVectorSearch
#from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
###from langchain.chains import RetrievalQAWithSourcesChain
import gradio as gr
from gradio.themes.base import Base

from dotenv import load_dotenv
import os

load_dotenv() # take env variables from .env file

# create the qdrant client for connecting to vector db
qdrant_url = os.getenv("qdrant_url")
print("qdrant_url : ", qdrant_url)
qdrant_client = QdrantClient(url=qdrant_url)
print("Qdrant Client created successfully")

# create the collection
collectionName = "collection_of_text_blobs_with_chunks_CS"

### START -- create openai embeddings
openai_api_key = os.getenv("openai_api_key")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
### END -- create openai embeddings

### START -- create the vector store
#vectorStore = MongoDBAtlasVectorSearch(collection, embeddings , index_name="rag_qna_vector_index_chunks")
vectorStore = Qdrant(
    embeddings = embeddings,
    client=qdrant_client,
    collection_name=collectionName
)
### END -- create the vector store

def query_data(query):
    print("query = ",query)
    ### Search the vector store with the query
    docs = vectorStore.similarity_search(query, k=1)
    print("Length of docs-->",len(docs))
    ### Get the first and the most relevant search result
    as_output = docs[0].page_content
    llm = OpenAI(openai_api_key=openai_api_key,temperature = 0, max_tokens=2048)
    print("llm created")
    retriever = vectorStore.as_retriever()
    print("retriever created")
    ###chain_type_kwargs = {"prompt": query}

    ###qa = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever = retriever, verbose=True,reduce_k_below_max_tokens=True)

    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever = retriever)
    print("attempting to start the query")
    retriever_output = qa.invoke(query)
    print("answer received")
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
demo.launch(share=True)


    
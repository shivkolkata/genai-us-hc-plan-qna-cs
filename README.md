# genai-us-hc-plan-qna-cs

# Installation Steps

1. Create python virtual environment - conda create -p venv python==3.9 -y

2. Install packages - pip install -r requirements.txt

3. Enter openai_api_key and qdrant_url in .env file (.env file need to be created)

4. Load Data, Convert to embeddings and insert into Mongo DB - python load_data.py

5. Navigate to Mongo DB Atlas collection, following collection should get created - rag_qna.collection_of_text_blobs_with_chunks

6. Create the index - refer mongodb-atlas-index.json file, Index name - rag_qna_vector_index_chunks

7. For testing - python extract_information.py

8. Open browser navigate to - http://127.0.0.1:7860 for entering question (sample question - "What is Medicare advantage ?")


import os
import streamlit as st
import PyPDF2
import textwrap
from pymilvus import Milvus, DataType

openai_api_key = st.secrets["OPENAI_API_KEY"]
zilliz_cloud_uri = st.secrets["ZILLIZ_CLOUD_URI"]
zilliz_cloud_api_key = st.secrets["ZILLIZ_CLOUD_API_KEY"]

def extract_text_from_pdfs(folder_loc):
    extracted_text = {}
    for file in os.listdir(folder_loc):
        text = ""
        with open(file, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        extracted_text[file] = text
    return extracted_text

def chunk_text(text, chunk_size=1000):
    return textwrap.wrap(text, chunk_size)

def generate_embeddings(text_chunks):
    from openai import OpenAI
    openai_client = OpenAI()
    embeddings = []
    for chunk in text_chunks:
        embedding = openai_client.embeddings.create(input=chunk, model="text-embedding-ada-002")
        embeddings.append(embedding['data'][0]['embedding'])
    return embeddings

def store_in_milvus(collection_name, extracted_text):
    client = Milvus()
    if not client.has_collection(collection_name):
        client.create_collection(collection_name, {
            "fields": [
                {"name": "id", "type": DataType.INT64, "is_primary": True},
                {"name": "vector", "type": DataType.FLOAT_VECTOR, "dim": 1536},
                {"name": "text", "type": DataType.VARCHAR, "max_length": 65535}
            ]
        })
    
    data = []
    id = 0
    for file, text in extracted_text.items():
        chunks = chunk_text(text)
        embeddings = generate_embeddings(chunks)
        for chunk, embedding in zip(chunks, embeddings):
            data.append({"id": id, "vector": embedding, "text": chunk})
            id += 1
    client.insert(collection_name, data)

def query_milvus(collection_name, query_text, top_k=5):
    client = Milvus()
    query_embedding = generate_embeddings([query_text])[0]
    results = client.search(collection_name, query_embedding, top_k=top_k)
    return results

def load_vector_database(collection_name):
    try:
        client = Milvus()
        if not client.has_collection(collection_name):
            st.error("Collection does not exist. Please process the PDF first.")
            return None
        return client
    except Exception as e:
        st.error(f"Failed to connect to the vector database: {e}")
        return None

st.title("PDF Text Extraction and Retrieval")

# Load the vector database at the start
collection_name = "my_rag_collection"
client = load_vector_database(collection_name)

if client is not None:
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files is not None:
        extracted_text = extract_text_from_pdfs("data")
        store_in_milvus(collection_name, extracted_text)
        st.success("PDFs processed and data stored in vector database.")

    query = st.text_input("Ask a question:")
    if query:
        results = query_milvus(collection_name, query)
        for result in results:
            st.write(result['text'])

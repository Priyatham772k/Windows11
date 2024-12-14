import os
import streamlit as st
import PyPDF2
import textwrap
from pymilvus import Milvus, DataType, connections
import openai

# Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
zilliz_cloud_uri = "https://in03-f2db10c5f456b31.serverless.gcp-us-west1.cloud.zilliz.com"
zilliz_cloud_api_key = "98807cbae03002ff10c5a6f14d3959c6dfad9a01127f351cbd0701e45b62751522a9e8acb409eb6f4fbcf6417595000b4df67623"

# Initialize Milvus
def initialize_milvus():
    try:
        connections.connect(
            alias="default",
            uri=zilliz_cloud_uri,
            token=zilliz_cloud_api_key
        )
        return True
    except Exception as e:
        st.error(f"Failed to connect to Milvus: {e}")
        return False

# Extract text from uploaded PDFs
def extract_text_from_uploaded_pdfs(uploaded_files):
    extracted_text = {}
    for uploaded_file in uploaded_files:
        text = ""
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        extracted_text[uploaded_file.name] = text
    return extracted_text

# Chunk text for embeddings
def chunk_text(text, chunk_size=1000):
    return textwrap.wrap(text, chunk_size)

# Generate OpenAI embeddings
def generate_embeddings(text_chunks):
    openai.api_key = openai_api_key
    embeddings = []
    for chunk in text_chunks:
        response = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
        embeddings.append(response['data'][0]['embedding'])
    return embeddings

# Store data in Milvus
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

# Query Milvus
def query_milvus(collection_name, query_text, top_k=5):
    client = Milvus()
    query_embedding = generate_embeddings([query_text])[0]
    results = client.search(collection_name, query_embedding, top_k=top_k)
    return results

# Streamlit App
st.title("PDF Text Extraction and Retrieval")

# Initialize Milvus
if not initialize_milvus():
    st.stop()

# Upload PDFs
uploaded_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
if uploaded_files:
    st.write("Processing uploaded PDFs...")
    extracted_text = extract_text_from_uploaded_pdfs(uploaded_files)

    if st.button("Store PDFs in Vector Database"):
        store_in_milvus("my_rag_collection", extracted_text)
        st.success("PDFs processed and stored in vector database.")

# Query the database
query = st.text_input("Ask a question:")
if query and st.button("Search"):
    results = query_milvus("my_rag_collection", query)
    for result in results:
        st.write(result['text'])

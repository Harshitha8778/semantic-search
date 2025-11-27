#RUN streamlit run c:/Users/user/OneDrive/Desktop/CodeInVSC/semantic/search.py IN TERMINAL
import fitz  # PyMuPDF
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient
import streamlit as st
import nltk
import os
from tempfile import NamedTemporaryFile

# Ensure NLTK tokenizer exists
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with fitz.open(pdf_file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to chunk text
def chunk_text(text, max_chunk_size=50):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Initialize Sentence Transformer
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.title("PDF Chunking and Searching App")

# File upload section
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    # Save the uploaded file to a temporary location
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())

    # Extract text from PDF
    pdf_text = extract_text_from_pdf(tmp_file.name)
    chunks = chunk_text(pdf_text)
    st.success("PDF Successfully Processed and Chunked!")

    # Display chunked text
    st.subheader("Chunks Extracted from PDF:")
    for idx, chunk in enumerate(chunks):
        st.write(f"Chunk {idx+1}:")
        st.write(chunk)

    # Initialize Qdrant client
    client = QdrantClient(
    url="https://8da9392c-1d67-41a7-bb12-ecc8da560b72.us-east-1-1.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ndwO-rP1EyrwoacQ_WRTnBCi5K3JnO3UsScXwGcbjR8")
    

    #api key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ndwO-rP1EyrwoacQ_WRTnBCi5K3JnO3UsScXwGcbjR8

    
    # Recreate collection and upload points
    client.recreate_collection(
        collection_name="mychunk",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )

    client.upload_points(
        collection_name="mychunk",
        points=[
            models.PointStruct(
                id=idx, vector=encoder.encode(chunk).tolist(), payload={"chunk": chunk}
            )
            for idx, chunk in enumerate(chunks)
        ],
    )

    # Function to search similar chunks
    def search_similar_chunks(query, limit=2):
        query_vector = encoder.encode(query).tolist()
        search_results = client.search(
            collection_name="mychunk",
            query_vector=query_vector,
            limit=limit
        )
        
        return search_results

    # Search query input
    query = st.text_input("Enter a search query:", "")

    # Search button
    if st.button("Search Similar Chunks"):
        # Search similar chunks
        results = search_similar_chunks(query)
        
        # Display search results
        st.subheader("Search Results:")
        for result in results:
          chunk_text = result.payload['chunk'].replace('\n', ' ')
          st.write(f"Similar Chunk: {chunk_text}, Score: {result.score}")


import os
import pandas as pd
from dotenv import load_dotenv

from datasets import load_dataset
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone, ServerlessSpec
from llama_index.llms.openai import OpenAI
from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.pinecone import PineconeVectorStore

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = "imdb-movies-index"
document_store = SimpleDocumentStore()
embeddings = OpenAIEmbedding(api_key=openai_api_key)
vector_store = PineconeVectorStore(index_name=index_name, embeddings=embeddings)

def download_data_and_create_embedding():
    # Create Pinecone vector store
    indexes = pc.list_indexes();
    index_names = [idx.name for idx in indexes]
    print(index_names)
    if index_name not in index_names:
        # Download an IMDB dataset from Hugging Face Hub, load the ShubhamChoksi/IMDB_Movies dataset
        dataset = load_dataset("ShubhamChoksi/IMDB_Movies")
        print(dataset)

        # Store imdb.csv from ShubhamChoksi/IMDB_Movies
        dataset_dict = dataset
        dataset_dict["train"].to_csv("imdb.csv")

        # Load CSV data
        csv_file_path = 'imdb.csv'
        df = pd.read_csv(csv_file_path)

        pc.create_index(name=index_name, dimension=1536, metric="cosine", spec=ServerlessSpec(cloud='aws', region='us-east-1')) 

        # Convert CSV rows to documents and add to the stores
        documents = []
        for index, row in df.iterrows():
            # Combine relevant columns into a single text (adjust as needed)
            text = ' '.join(row.astype(str).values)
            doc = Document(text=text)
            documents.append(doc)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # create index, which will insert documents/vectors to pinecone
        vector_index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        print(vector_index)
        print("Documents added to Pinecone index")
        index = pc.Index(index_name)
        print(index.describe_index_stats())

# Function to retrieve relevant documents and generate text
def retrieve_and_generate(query, vector_store):
    index = pc.Index(index_name)
    
    # Retrieve vectors from Pinecone (example: all vectors)
    vector_store = PineconeVectorStore(index_name=index_name)
    print(vector_store)
    # Create VectorStoreIndex
    vector_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embeddings)
    llm = OpenAI(temperature=0, model="gpt-4o-2024-05-13")

    Settings.llm = llm
    Settings.chunk_size = 1024

    # question we ask the chat model
    query_str = "What are some good sci-fi movies from the 1980s?"

    query_engine = vector_index.as_query_engine(llm=llm)
    response = query_engine.retrieve(query)

    return response

# # Call the function to download data and create embeddings
# download_data_and_create_embedding()

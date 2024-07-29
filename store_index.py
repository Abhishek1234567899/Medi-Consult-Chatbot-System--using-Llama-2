from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

PINECONE_API_KEY = "63650804-25eb-4a2b-979c-83b11d35f0e9"
PINECONE_API_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medicine"

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# Check if the index exists, and create it if it doesn't
if PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes().index_list["indexes"]]:
    pc.create_index(
        name=PINECONE_INDEX_NAME, 
        dimension=384,  # Replace with the actual dimension of your embeddings
        metric='cosine',  # Or the appropriate metric for your use case
        spec=ServerlessSpec(
            cloud='aws',
            region=PINECONE_API_ENV
        )
    )

# Create PineconeVectorStore from texts
vector_store = PineconeVectorStore.from_texts(
    [chunk.page_content for chunk in text_chunks],
    embeddings,
    index_name=PINECONE_INDEX_NAME
)

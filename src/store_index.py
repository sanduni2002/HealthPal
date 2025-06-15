from src.helper import load_pdf_file,text_split,download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

#PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
#os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file(data='/Users/sanu/Documents/Projects/HealthPal/HealthPal/Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pinecone_key = 'pcsk_2Aq24s_RqWmnwrzNHpXm39o5JyKauAUuWxuzZen5c8Mh8jTkZnGBt2obsQQaSTTcwjY7iL'
pc = Pinecone(api_key=pinecone_key)


index_name = "healthpal"

pc.create_index(
    name = index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

#Embed each chunk and upsert the embeddings into your Pinecone index
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name = index_name,
    embedding = embeddings,
)
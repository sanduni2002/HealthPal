import traceback
from flask import Flask, render_template, jsonify, request, send_from_directory
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from pinecone import Pinecone

app = Flask(__name__)

# Initialize Pinecone client first
PINECONE_API_KEY = "pcsk_2Aq24s_RqWmnwrzNHpXm39o5JyKauAUuWxuzZen5c8Mh8jTkZnGBt2obsQQaSTTcwjY7iL"
MISTRALAI_API_KEY = "tdaJUi5tqeX086zUS6XBC5lg3DSil31K"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

embeddings = download_hugging_face_embeddings()

index_name = "healthpal"

# Set the API key for LangChain compatibility
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

from langchain_mistralai import ChatMistralAI
llm = ChatMistralAI(
    model="mistral-large-latest",
    mistral_api_key=MISTRALAI_API_KEY,
    temperature=0.4,
    max_tokens=500
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET","POST"])
def chat():
    try:
        # Debug headers and form data
        print("\n=== Incoming Request ===")
        print("Headers:", request.headers)
        print("Form data:", request.form)
        
        msg = request.form.get("msg")
        if not msg or msg.strip() == "":
            print("Empty message received")
            return "Please enter a valid message", 400

        print(f"Processing message: {msg}")
        
        response = rag_chain.invoke({"input": msg})
        answer = response["answer"]
        
        if not answer or answer.strip() == "":
            print("Empty response generated")
            return "I didn't understand that. Please try again.", 200
            
        print(f"Generated answer: {answer}")
        return answer
        
    except Exception as e:
        print(f"\n!!! Error Processing Message !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        traceback.print_exc()
        return "I encountered an error processing your request", 500
@app.route("/ping")
def ping():
    return jsonify({"status": "success", "message": "Backend is reachable"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
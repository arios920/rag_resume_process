from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
import PyPDF2
import os
import pandas as pd
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAI

app = FastAPI()

class Message(BaseModel):
    text: str

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up the OpenAI client
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

resume_text = ""
chat_history = []
index = None

# Load data from jobs.csv
df = pd.read_csv("jobs.csv")

# Combine relevant columns into a single text field for each job listing
df['combined_text'] = df.apply(lambda row: f"title: {row['title']} company: {row['company']} Industries: {row['Industries']} Seniority level: {row['Seniority level']} description: {row['description']} education: {row['education']} months_experience: {row['months_experience']} salary: {row['salary']}", axis=1)

texts = df['combined_text'].tolist()

# Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Encode the texts
embeddings = embedding_model.embed_documents(texts)
embeddings = np.array(embeddings)

# Initialize FAISS vector store
vector_store = FAISS.from_texts(texts, embedding_model, ids=df.index.tolist())

# OpenAI LLM
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# RetrievalQA Chain
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
)

@app.get("/", response_class=HTMLResponse)
async def get():
    with open("templates/index.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...)):
    global resume_text
    global index
    try:
        # Save uploaded file
        file_location = f"temp/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        
        # Read PDF file
        with open(file_location, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            resume_text = ""
            for page in range(len(reader.pages)):
                resume_text += reader.pages[page].extract_text()
        
        # Load document and create index
        loader = PyPDFLoader(file_location)
        documents = loader.load()
        embeddings = OpenAIEmbeddings()
        index = VectorstoreIndexCreator(vectorstore_cls=FAISS, embedding=embeddings).from_documents(documents)

        return JSONResponse(content={"message": "Resume uploaded and indexed successfully."})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/chat/")
async def chat(message: Message):
    global resume_text
    global chat_history
    global index

    if resume_text == "":
        return JSONResponse(content={"answer": "Please upload your resume first."})
    
    # Perform RAG
    question = message.text
    query = f"Resume Information: {resume_text} Question: {question}"
    response = retrieval_qa_chain({"query": query})

    return JSONResponse(content={"answer": response["result"]})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import datetime
import pinecone

from typing import List, Dict

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferMemory

from langchain.schema.document import Document

from utils import get_filename_from_filekey
from utils import upload_to_s3
from utils import get_data_from_s3
from utils import get_list_from_filedata
from utils import get_list_from_textdata

from utils import get_summary

from utils import get_extract_single
from utils import get_extract_parallel

from utils import build_documents_list
from utils import query_selected_docs
from utils import build_laws_list
from utils import query_selected_laws

from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CHUNK_SIZE = os.environ.get("CHUNK_SIZE")

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_DIRECTORY = os.environ.get("S3_DIRECTORY")

INDEX_NAME = os.environ.get("INDEX_NAME")

os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

pinecone.init(
  api_key=PINECONE_API_KEY, 
  environment=PINECONE_ENV, 
)

app = FastAPI()

origins = [
  "http://localhost",
]
app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

@app.post("/create_index")
async def create_index( # Create a new index in Pinecone 
  index_name: str = Form(...)
):
  try:
    if index_name not in pinecone.list_indexes():
      pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536  
      )
      return {"result": "success"}
  except:
    return {"result": "fail"}

@app.post("/delete_index")
async def delete_index( # Delete an index in Pinecone
  index_name: str = Form(...)
):
  try:
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
        return {
          "result": "success"
        }
  except:
    return {
      "result": "fail"
    }
  
@app.post("/upload_pdf")
async def upload_pdf( # Upload pdf file to AWS S3 bucket
  file: UploadFile = File(...), 
  user_id: int = Form(...), 
  project_id: int = Form(...)
):
  current_datetime = datetime.datetime.now()
  formatted_time = current_datetime.strftime("%y%m%d_%H%M%S")
  prefix = f"{user_id}_{project_id}_{formatted_time}_"
  uploaded_file_name = f"{prefix}{file.filename}"
  success = upload_to_s3(file, S3_BUCKET_NAME, f"{S3_DIRECTORY}{prefix}")

  if success:
      return {"result": uploaded_file_name}
  else:
      return {"result": "Failed to upload file to S3"}
  
@app.post("/save_doc_to_vector")
async def save_to_vectordb( # Save document to Vector DB(Pinecone)
  index_name: str = Form(...),
  bucket: str = Form(...), 
  file_key: str = Form(...), 
  document_id: int = Form(...), 
  project_id: int = Form(...), 
  type: str = Form(...), 
  source: str = Form(...), 
  connector_type: str = Form(...)
):
  try:
    meta_info = {
      "category": "document",
      "document_id": document_id,
      "project_id": project_id,
      "type": type,
      "source": source,
      "connector_type": connector_type,
      "file_key": file_key,
      "file_name": get_filename_from_filekey(file_key)
    }

    file_data = get_data_from_s3(bucket, f"{file_key}")
    document_list = get_list_from_filedata(file_data, int(CHUNK_SIZE), meta_info)

    index = pinecone.Index(index_name)
    vectorstore = Pinecone(index, embeddings, "text")
    
    for doc in document_list:
      vectorstore.add_documents([doc])

    return {
      "result": "success"
    }
  except Exception as e:
    return {
      "result": str(e)
    }

@app.post("/save_law_to_vector")
async def save_law_to_vector( # Save law text to Vector DB(Piencone)
  index_name: str = Form(...), 
  law_text: str = Form(...), 
  article_number: str = Form(...), 
  code: str = Form(...), 
  date_enforced: int = Form(...), 
  date_cancelled: int = Form(...), 
  juridiction: str = Form(...)
):
  try:
    meta_info = {
      "category": "law",
      "article_number": article_number,
      "code": code,
      "date_enforced": date_enforced,
      "date_cancelled": date_cancelled,
      "juridiction": juridiction
    }
    
    document_list = get_list_from_textdata(law_text, int(CHUNK_SIZE), meta_info)
    
    index = pinecone.Index(index_name)
    vectorstore = Pinecone(index, embeddings.embed_query, "text")
    vectorstore.add_documents(document_list)

    return {
      "result": "success" 
    }
  except:
    return {
      "result": "failed!"
    }

@app.post("/query_law")
async def query_law(
  user_id:int, 
  project_id:int, 
  query_law_id:int, 
  query_law_name:str, 
  query_law_custom_request:str, 
  top_k: int,
  laws_source:list[str], 
  laws_type:list[str], 
  laws_list:list[str],
  laws_table: list[dict],
  llm_memory: list[dict]
):
  law_keys = build_laws_list(user_id, project_id, laws_source, laws_type, laws_list, laws_table)
  result = query_selected_laws(INDEX_NAME, query_law_custom_request, top_k, law_keys)
  new_llm_memory = llm_memory
  new_llm_memory.append({"question": query_law_custom_request, "answer": result["gpt_answer"]})
  return {
    "user_id": user_id,
    "project_id": project_id,
    "query_law_id": query_law_id,
    "query_law_custom_request": query_law_custom_request,
    "query_answer": result["gpt_answer"],
    "relevant_laws": result["ref_data"],
    "llm_memory": new_llm_memory,
    "laws_list": laws_list
  } 

@app.post("/query_project")
async def query_project(
  user_id:int, 
  project_id:int, 
  query_id:int, 
  query_name:str, 
  query_custom_request:str, 
  top_k: int,
  docs_source:list[str], 
  docs_type:list[str], 
  docs_list:list[str], 
  docs_table:list[dict],
  llm_memory: list[dict]
):
  file_keys = build_documents_list(user_id, project_id, None, docs_source, docs_type, docs_list, docs_table)
  result = query_selected_docs(INDEX_NAME, query_custom_request, top_k, file_keys)
  new_llm_memory = llm_memory
  new_llm_memory.append({"question": query_custom_request, "answer": result["gpt_answer"]})
  return {
    "user_id": user_id,
    "project_id": project_id,
    "query_id": query_id,
    "query_custom_request": query_custom_request,
    "query_answer": result["gpt_answer"],
    "relevant_docs": result["ref_data"],
    "llm_memory": new_llm_memory,
    "docs_list": docs_list
  }

@app.post("/summarize_documents")
async def summarize_documents(
  user_id:int, 
  project_id:int, 
  summary_id:int, 
  summary_name:str, 
  summary_approach:str, 
  summary_custom_request:str, 
  docs_source:list[str], 
  docs_type:list[str], 
  docs_list:list[str], 
  docs_table:list[dict]
):
  file_keys = build_documents_list(user_id, project_id, None, docs_source, docs_type, docs_list, docs_table)
  final_result = get_summary(summary_approach, summary_custom_request, file_keys)
  return {
    "user_id": user_id,
    "project_id": project_id,
    "summary_id": summary_id,
    "summary_name": summary_name,
    "summary_approach": summary_approach,
    "summary_custom_request": summary_custom_request,
    "summary": final_result
  }

@app.post("/extract_data")
async def extract_data(
  user_id:int, 
  project_id:int, 
  extract_id:int, 
  extract_name:str, 
  extract_custom_request:str,
  extract_approach:list[str],  
  docs_source:list[str], 
  docs_type:list[str], 
  docs_list:list[str], 
  docs_table:list[dict]
):
  file_keys = build_documents_list(user_id, project_id, None, docs_source, docs_type, docs_list, docs_table)
  final_result = get_extract_parallel(
    user_id,
    project_id,
    extract_id,
    extract_approach,
    extract_custom_request,
    docs_table,
    file_keys,
    get_extract_single
  )
  return final_result

if __name__ == "__main__":
  uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)



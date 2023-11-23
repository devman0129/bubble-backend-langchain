from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import S3FileLoader
from langchain.schema.document import Document
import concurrent.futures
import boto3
import os
import pinecone
import json
import tiktoken
from botocore.exceptions import NoCredentialsError
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import create_extraction_chain
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from typing import List, Dict

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

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def get_filename_from_filekey(file_key): # Get file name from filekey of AWS S3 bucket
  array = file_key.split("_")
  file_name = array[-1]
  return file_name

def upload_to_s3(file, bucket_name, s3_directory):
  s3 = boto3.client('s3')
  try:
      s3.upload_fileobj(
          file.file,
          bucket_name,
          s3_directory + file.filename
      )
      return True
  except NoCredentialsError:
      return False

def get_data_from_s3(bucket, file_key): # Get file data from a file of AWs S3 bucket
  loader = S3FileLoader(bucket, file_key)
  return loader.load()

def get_list_from_filedata(file_data, chunk_size, meta_info): # Get a list of file data by splitting into chunks
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
  texts = text_splitter.split_documents(file_data)
  document_list = []
  for text in texts:
    metadata = meta_info
    document = Document(
      page_content=text.page_content,
      metadata= metadata
    )
    document_list.append(document)
  return document_list
  
def get_list_from_textdata(text_data, chunk_size, meta_info): # Get a list of text data by splitting into chunks
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
  texts = text_splitter.split_text(text_data)
  document_list = []
  for text in texts:
    document = Document(
      page_content=text,
      metadata= meta_info
    )
    document_list.append(document)
  return document_list

def get_relevant_data(index, embeddings, input, filter_condition, top_k): # Get relevant data by similarity search
  input_vector = embeddings.embed_query(input)
  result = index.query(
    vector=input_vector,
    filter=filter_condition,
    top_k=top_k,
    include_metadata=True
  )
  res_list = result.to_dict()
  final_result = []
  for item in res_list["matches"]:
    final_result.append(item["metadata"])
  return final_result

def extract_data_by_approach(bucket, file_key, custom_approach):
    file_data = get_data_from_s3(bucket, f"{file_key}")
    document_list = get_list_from_filedata(file_data, 5000, {})
    schema = {
      "properties": {
          "Dates": {"type": "string"},
          "Places": {"type": "string"},
          "Names": {"type": "string"},
          "Events": {"type": "string"},
          "Entities": {"type": "string"},
      },
      "required": custom_approach,
    }
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    chain = create_extraction_chain(schema, llm)
    total_result = {}
    for doc in document_list:
      res = chain.run(doc.page_content)
      for key, value in res[0].items():
        if key in total_result:
          total_result[key] += value
        else:
          total_result[key] = value
    filtered_data = {key: total_result[key] for key in custom_approach if key in total_result}
    return filtered_data

def extract_data_by_custom(bucket, file_key, custom_request):
  try:
    file_data = get_data_from_s3(bucket, f"{file_key}")
    document_list = get_list_from_filedata(file_data, 5000, {})
    total_result = ""
    for doc in document_list:   
      prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant."), # The persistent system prompt
        MessagesPlaceholder(variable_name="chat_history"), # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template("{human_input}"), # Where the human input will injectd
      ])
      memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
      llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
      chat_llm_chain = LLMChain(
          llm=llm,
          prompt=prompt,
          verbose=True,
          memory=memory,
      )
      input = f"{custom_request} \n Get the result as json schema. content: {doc.page_content}"
      result = chat_llm_chain.predict(human_input=input)
      total_result += result
    return total_result

  except:
    return "failed!"

def summarize(doc_list, request, type):
  try:
    prompts = {
      "short": "Please summarize the following content using 50 words.",
      "medium": "Please summarize the following content using 200 words.",
      "detailed": "Please summarize following the content using 500 words.",
    }
    summary = ""
    for doc in doc_list:   
      prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant."), # The persistent system prompt
        MessagesPlaceholder(variable_name="chat_history"), # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template("{human_input}"), # Where the human input will injectd
      ])
      memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
      llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
      chat_llm_chain = LLMChain(
          llm=llm,
          prompt=prompt,
          verbose=True,
          memory=memory,
      )
      input = ""
      if type == 1:
        input = f"{prompts[request]} content: {summary}\n{doc.page_content}"
      else:
         input = f"{request} content: {summary}\n{doc.page_content}"
      result = chat_llm_chain.predict(human_input=input)
      summary = result
    return summary

  except:
    return "failed!" 
def summarize_one_doc(bucket, file_key, request, type):
  try:
    prompts = {
      "short": "Please summarize the following content using 50 words.",
      "medium": "Please summarize the following content using 200 words.",
      "detailed": "Please summarize following the content using 500 words.",
    }
    file_data = get_data_from_s3(bucket, f"{file_key}")
    document_list = get_list_from_filedata(file_data, 5000, {})
    summary = ""
    for doc in document_list:   
      prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant."), # The persistent system prompt
        MessagesPlaceholder(variable_name="chat_history"), # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template("{human_input}"), # Where the human input will injectd
      ])
      memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
      llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
      chat_llm_chain = LLMChain(
          llm=llm,
          prompt=prompt,
          verbose=True,
          memory=memory,
      )
      input = ""
      if type == 1:
        input = f"{prompts[request]} content: {summary}\n{doc.page_content}"
      else:
         input = f"{request} content: {summary}\n{doc.page_content}"
      result = chat_llm_chain.predict(human_input=input)
      summary = result
    return summary

  except:
    return "failed!"

def process_in_parallel(bucket: str, file_list: List[str], func):
    results = []
    files = file_list[0].split(",")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(files)) as executor:
        future_to_file = {executor.submit(func, bucket, file): file for file in files}
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                response = future.result()
                results.append({"file_key": file, "result": response})
            except Exception as e:
                print(f"Error processing file '{file}': {str(e)}")
    return results

def num_tokens_from_string(string: str) -> int:
    encoding_name = "cl100k_base"
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_gpt_answer(input_query: str, ref_data: str):
  try:
    prompt = ChatPromptTemplate.from_messages([
      SystemMessage(content="You are a helpful assistant."), 
      MessagesPlaceholder(variable_name="chat_history"), 
      HumanMessagePromptTemplate.from_template("{human_input}"),
    ])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    chat_llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    input = f"{input_query} You can use the following information. information: {ref_data}"
    answer = chat_llm_chain.predict(human_input=input)
    return answer
  except:
    return "failed!"

def build_documents_list(
    user_id: int,
    project_id: int,
    docs_scope: str,
    docs_source: List[str],
    docs_type: List[str],
    docs_list: List[str],
    docs_table: List[Dict]
) -> List[str]:
    if docs_list:
        file_keys = [doc['file_key'] for doc in docs_table if doc['doc_name'] in docs_list and doc['user_id'] == user_id and doc['project_id'] == project_id]
    elif docs_scope:
        if docs_scope in ['Client', 'Opponent', 'Others']:
            file_keys = [doc['file_key'] for doc in docs_table if doc['doc_type'] == docs_scope and doc['user_id'] == user_id and doc['project_id'] == project_id]
        else:
            file_keys = [doc['file_key'] for doc in docs_table if doc['doc_source'] == docs_scope and doc['user_id'] == user_id and doc['project_id'] == project_id]
    else:
        file_keys = [
            doc['file_key'] for doc in docs_table
            if doc['doc_source'] in docs_source and doc['doc_type'] in docs_type and doc['user_id'] == user_id and doc['project_id'] == project_id
        ]

    return file_keys

def filter_ref_data(ref_data):
   text = ""
   for data in ref_data:
      text += data["text"] + "\n"
   return text 


def query_selected_docs(index_name, query_custom_request, top_k, file_keys):
  index = pinecone.Index(index_name)
  filter_condition = {
    "file_key": {"$in":file_keys}
  }
  ref_data = get_relevant_data(index, embeddings, query_custom_request, filter_condition, top_k)
  num_tokens = num_tokens_from_string(query_custom_request + filter_ref_data(ref_data))
  if num_tokens <= 8000:
    gpt_answer = get_gpt_answer(query_custom_request, filter_ref_data(ref_data))
  else:
     gpt_answer = "token limited"
  return {
     "ref_data": ref_data,
     "gpt_answer": gpt_answer
  }

def build_laws_list(
    user_id: int,
    project_id: int,
    laws_source: List[str],
    laws_type: List[str],
    laws_list: List[str],
    laws_table: List[Dict]
) -> List[str]:
    if laws_list:
        law_keys = [law['article_number'] for law in laws_table if law['code'] in laws_list]
    else:
        law_keys = [
            law['article_number'] for law in laws_table
            if law['law_source'] in laws_source and law['law_type'] in laws_type
        ]

    return law_keys

def query_selected_laws(index_name, query_custom_request, top_k, law_keys):
  index = pinecone.Index(index_name)
  filter_condition = {
    "article_number": {"$in":law_keys}
  }
  ref_data = get_relevant_data(index, embeddings, query_custom_request, filter_condition, top_k)
  num_tokens = num_tokens_from_string(query_custom_request + json.dumps(ref_data))
  if num_tokens <= 8000:
    gpt_answer = get_gpt_answer(query_custom_request, json.dumps(ref_data))
  else:
     gpt_answer = "token limited"
  return {
     "ref_data": ref_data,
     "gpt_answer": gpt_answer
  }

def get_summary(summary_approach, summary_custom_request, file_list):
  doc_list = []
  for file_key in file_list:
    file_data = get_data_from_s3(S3_BUCKET_NAME, f"{file_key}")
    element_list = get_list_from_filedata(file_data, 5000, {})
    doc_list.extend(element_list)
  print(doc_list)
  if summary_approach == "short" or summary_approach == "medium" or summary_approach == "detailed":
    summarize_result = summarize(doc_list, summary_approach, 1) 
  else:
    summarize_result = summarize(doc_list, summary_custom_request, 2)
  return summarize_result

def get_extract_single(extract_approach, extract_custom_request, file_key):
  extract_answer = ""
  if len(extract_approach) > 0:
    extract_answer = extract_data_by_approach(S3_BUCKET_NAME, file_key, extract_approach)
  else:
    extract_answer = extract_data_by_custom(S3_BUCKET_NAME, file_key, extract_custom_request)
  return extract_answer

def get_extract_parallel(
  user_id,
  project_id,
  extract_id,
  extract_approach,
  extract_custom_request,
  docs_table,
  file_list,
  func
):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(file_list)) as executor:
        future_to_file = {executor.submit(func, extract_approach, extract_custom_request, file): file for file in file_list}
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                response = future.result()

                doc_id = ""
                doc_name = ""

                for doc in docs_table:
                  if doc["file_key"] == file:
                    doc_id = doc["doc_id"]
                    doc_name = doc["doc_name"]
                    break
                
                extract = {
                  "user_id": user_id,
                  "project_id": project_id,
                  "extract_id": extract_id,
                  "doc_id": doc_id,
                  "doc_name": doc_name,
                  "extract_approach": extract_approach,
                  "extract_custom_request": extract_custom_request,
                  "extract_answer": response
                }

                results.append(extract)
            except Exception as e:
                print(f"Error processing file '{file}': {str(e)}")
    return results
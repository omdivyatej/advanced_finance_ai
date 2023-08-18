import os
from dotenv import load_dotenv
import openai
import logging

# Import functions from the separate files
from utils.external_files_loading import loadweb_document
from utils.text_splitter import split_document
from embeddings_utils.embeddings_manager import manage_embeddings
from utils.chat_setup import setup_chat
from utils.prompts import chat_template

#Langchain Imports
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuration Constants
URL = "https://frontiernerds.com/files/state_of_the_union.txt"
PATH_TO_EMBEDDINGS = "embeddings_utils/embeddings_db"
TEMPERATURE_LLM = 0
LLM_MODEL_NAME = "gpt-3.5-turbo"

logging.basicConfig(level=logging.INFO)

# Load the document
raw_document_name, raw_documents = loadweb_document(URL)

# Split the document
chunked_doc_splits = split_document(raw_documents)

# Manage embeddings
store_name = raw_document_name
db = manage_embeddings(store_name, PATH_TO_EMBEDDINGS, chunked_doc_splits)

if db is None:
    logging.error("Failed to manage embeddings. Exiting.")
    exit()

# Build a chain
llm = ChatOpenAI(temperature=TEMPERATURE_LLM, verbose=True, model_name=LLM_MODEL_NAME)

# Set up the history prompt template
history_prompt_template = PromptTemplate(
    template=chat_template, input_variables=["history", "context", "question"]
)

# Set up the chat
retriever = db.as_retriever()
chat = setup_chat(llm, retriever, history_prompt_template)

def process_user_query(chat):
    while True:
        query = input("Ask a question, or type 'exit' to quit:  ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        result = chat.run(query)
        print(result)

process_user_query(chat)

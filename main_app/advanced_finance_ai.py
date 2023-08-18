#Project Imports
import os
from dotenv import load_dotenv
from embeddings import load_embeddings,store_embeddings
from urllib.parse import urlparse

#AI Imports
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
'''
FO- Fairly optimised
CO - Can be optimised more
BO - Badly optimised
'''


# Importing ENV variables
load_dotenv()
openai.api_key= os.getenv("OPENAI_API_KEY")

#Load the document - FO 

url="https://frontiernerds.com/files/state_of_the_union.txt"
parsed_url = urlparse(url)
loader = WebBaseLoader(url,continue_on_failure=True, verify_ssl=True)
raw_document_name = str(parsed_url.netloc)+"_"+str(parsed_url.path).replace("/","_") 
raw_documents= loader.load()

#Split the document - Recursive is recommended due to iterative approach
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50,length_function=len)
chunked_doc_splits = text_splitter.split_documents(raw_documents)

#Store the splits - use the embeddings 
# Check if embeddings for the given file/web are already stored
store_name = raw_document_name  # Replace with an identifier for your file/web

path_to_embeddings = "embeddings_db"  # Replace with the actual path
try:
    loaded_embeddings = load_embeddings(store_name, path_to_embeddings)
    print(loaded_embeddings)

    if loaded_embeddings is None:
        # If embeddings are not already stored, then store them       
        stored_emb = store_embeddings(docs=chunked_doc_splits, embeddings=OpenAIEmbeddings(), store_name=store_name, path=path_to_embeddings)
        print('storing the emb')
        db=stored_emb
    else:
        # If embeddings are already stored, use them
        db = loaded_embeddings
        print('retrieving the emb')
except Exception as e:
    print("Error:", e)


#Build a chain
llm=ChatOpenAI(temperature=0,verbose=True,model_name="gpt-3.5-turbo")

#set the retriever
retriever = db.as_retriever()

sales_template = """
     As an excellent history and general knowledge bot, your goal is to provide accurate and helpful information
     about the context provided to you. You should answer user inquiries based on the context provided.
     If he greets, then greet him. Don't include prefix 'Answer'. If you don't understand a question, ask to 
     repeat the question. If you see a question related to anything irrelevant to the context, say it is irrelevant.
     Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    <ctx>
     {context} 
    </ctx>
    <hs> {history} </hs>
    Question: {question}"""
SALES_PROMPT = PromptTemplate(
        template=sales_template, input_variables=["history", "context", "question"]
    )
chat = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever= db.as_retriever(),
    chain_type_kwargs={
            "verbose": False,
            "prompt": SALES_PROMPT,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"),
        },
        verbose = True

)

#Ask the question
while True:
    query = input("Ask a question, or type 'exit' to quit:  ")
    if query.lower()=='exit':
        break

    result=chat.run(query)
    print(result)

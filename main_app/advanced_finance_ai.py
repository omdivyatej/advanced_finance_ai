#Project Imports
import os
from dotenv import load_dotenv
from embeddings import load_embeddings,store_embeddings
from urllib.parse import urlparse,parse_qs

#AI Imports
import openai
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

'''
FO- Fairly optimised
CO - Can be optimised more
BO - Badly optimised
'''


# Importing ENV variables
load_dotenv()
openai.api_key= os.getenv("OPENAI_API_KEY")

#Load the document - FO 
url= "https://lilianweng.github.io/posts/2023-06-23-agent/"
parsed_url = urlparse(url)
loader = WebBaseLoader(url,continue_on_failure=True, verify_ssl=True)
raw_document_name = str(parsed_url.netloc)+"_"+str(parsed_url.path).replace("/","_")
raw_documents= loader.load()

#Split the document - Recursive is recommended due to iterative approach
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50,length_function=len)
chunked_doc_splits = text_splitter.split_documents(raw_documents)

#Store the splits - use the embeddings 
# Check if embeddings for the given file/web are already stored
store_name = "om"  # Replace with an identifier for your file/web

path_to_embeddings = "embeddings_db"  # Replace with the actual path
try:
    loaded_embeddings = load_embeddings(store_name, path_to_embeddings)
    print(loaded_embeddings)

    if loaded_embeddings is None:
        # If embeddings are not already stored, then store them
        #db = FAISS.from_documents(chunked_doc_splits, embedding=OpenAIEmbeddings())
        store_embeddings(docs=chunked_doc_splits, embeddings=OpenAIEmbeddings(), store_name=store_name, path=path_to_embeddings)
        print('storing the emb')
    else:
        # If embeddings are already stored, use them
        db = loaded_embeddings
        print('retrieving the emb')
except Exception as e:
    print("Error:", e)

#db = FAISS.from_documents(chunked_doc_splits, embedding=OpenAIEmbeddings())

# #Build a chain
# llm=OpenAI(temperature=0)
# chain = load_qa_chain(llm, chain_type="stuff")


# #Ask the question
# while True:
#     query = input("Ask a question, or type 'exit' to quit:  ")
#     if query.lower()=='exit':
#         break

#     #question = "What are the approaches to Task Decomposition?"
#     answer_docs = db.similarity_search(query)
#     op = chain.run(input_documents=answer_docs,question=query)
#     print(op)
#     # where the part of the answer is found
#     print(answer_docs[0].page_content)



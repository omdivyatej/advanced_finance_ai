import pickle
import faiss
from langchain.vectorstores import FAISS
import os

def store_embeddings(docs, embeddings, store_name, path):
    vectorStore = FAISS.from_documents(docs, embeddings)    
    file_path = f"{path}/faiss_{store_name}.pkl"
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            pickle.dump(vectorStore, f)
        print("Embeddings stored successfully.")
    else:
        print("Embeddings file already exists. Not overwriting. Emb error")

def load_embeddings(store_name, path):
    file_path = f"{path}/faiss_{store_name}.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            VectorStore = pickle.load(f)
        return VectorStore
    else:
        return None
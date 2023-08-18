from embeddings_utils.embeddings import load_embeddings, store_embeddings
from langchain.embeddings import OpenAIEmbeddings

def manage_embeddings(store_name, path_to_embeddings, chunked_doc_splits):
    try:
        loaded_embeddings = load_embeddings(store_name, path_to_embeddings)
        if loaded_embeddings is None:
            stored_emb = store_embeddings(
                docs=chunked_doc_splits, embeddings=OpenAIEmbeddings(), store_name=store_name, path=path_to_embeddings
            )
            db = stored_emb
        else:
            db = loaded_embeddings
        return db
    except Exception as e:
        print("Error:", e)
        return None

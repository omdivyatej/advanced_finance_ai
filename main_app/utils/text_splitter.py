from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_document(raw_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    chunked_doc_splits = text_splitter.split_documents(raw_documents)
    return chunked_doc_splits

from langchain.document_loaders import WebBaseLoader
from urllib.parse import urlparse

def loadweb_document(url):
    parsed_url = urlparse(url)
    loader = WebBaseLoader(url, continue_on_failure=True, verify_ssl=True)
    raw_document_name = str(parsed_url.netloc) + "_" + str(parsed_url.path).replace("/", "_")
    raw_documents = loader.load()
    return raw_document_name, raw_documents
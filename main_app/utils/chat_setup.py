from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

def setup_chat(llm, retriever, history_prompt_template):
    chat = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "verbose": False,
            "prompt": history_prompt_template,
            "memory": ConversationBufferMemory(memory_key="history", input_key="question"),
        },
        verbose=True
    )
    return chat

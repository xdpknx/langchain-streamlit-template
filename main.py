"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import pickle
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
        print(vectorstore)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        chain = ConversationalRetrievalChain.from_llm(llm=llm,
            retriever=vectorstore.as_retriever(), verbose=True, return_source_documents=False)
    return chain

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("Botswana.AI")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "history" not in st.session_state:
    st.session_state["history"] = []

def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chain( {"question": user_input, "chat_history": st.session_state["history"]})

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output["answer"])

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

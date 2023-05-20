"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import pickle
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from PIL import Image


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
        llm = OpenAI( temperature=0)
        question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
        doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff")
        chain = ConversationalRetrievalChain(
    retriever=vectorstore.as_retriever(),
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
)
    return chain



# From here down is all the StreamLit UI.
st.set_page_config(page_title="Botswana AI Citizen", page_icon=":robot:")
st.header("Botswana.AI")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "history" not in st.session_state:
    st.session_state["history"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

#image = Image.open("Code-of-Arms-colour.png")
#st.image(image, caption='Your Image', use_column_width=True)

chain = load_chain()
image_placeholder = st.empty()

user_input = get_text()

if user_input:
    with st.spinner('Finding answers'):
        output = chain( {"question": user_input, "chat_history": st.session_state["history"]})
    st.success('Done!')

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output["answer"])

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

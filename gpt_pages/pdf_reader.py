import streamlit as st
import time
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import HumanMessage

from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


from modules.my_gpt.model import get_model
from modules.my_gpt.retriever import create_retriever

# 채팅 히스토리 초기화
if "store" not in st.session_state:
    st.session_state.store = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

# 업로드한 PDF 파일 데이터(컨텍스트) 초기화
if "_pdf_pages" not in st.session_state:
    st.session_state._pdf_pages = []

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.makedirs(".cache/files")

# 업로드한 PDF 파일 경로
if "_uploaded_pdf_file_path" not in st.session_state:
    st.session_state._uploaded_pdf_file_path = None

# 세션 정보
if "session_config" not in st.session_state:
    st.session_state.session_config = {"configurable": {"session_id": "abc2"}}


# 세션 아이디(session_id)를 받아서 메세지 히스토리 오브젝트를 반환
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def response_generator(response):
    for word in response["answer"].split():
        yield word + " "
        time.sleep(0.05)


# RAG 체인 생성
def create_rag_chain():
    # 모델 생성
    llm = get_model(
        model_id=st.session_state._model_id,
        model_api_key=st.session_state._model_api_key,
    )

    # 검색자 생성
    retriever = create_retriever(
        st.session_state._uploaded_pdf_file_path, 
        st.session_state._model_api_key
    )

    # 시스템 프롬프트 생성
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # retriever 체인 생성
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    # conversational rag 체인 생성
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain


# GPT 채팅 응답 생성
def generate_response(human_message):
    chain = create_rag_chain()  # RAG 체인 생성

    return response_generator(
        chain.invoke(
            {"input": human_message}, config=st.session_state.session_config
        )
    )


# PDF 파일 업로드 핸들러
def upload_pdf_file():
    file = st.session_state.uploaded_pdf_file

    if file is not None:
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)

        # 업로드한 파일 경로를 세션 정보에 저장
        st.session_state._uploaded_pdf_file_path = file_path
    else:
        # 업로드한 파일 삭제
        if os.path.exists(st.session_state._uploaded_pdf_file_path):
            os.remove(st.session_state._uploaded_pdf_file_path)
        st.session_state._uploaded_pdf_file_path = None 


# 앱 타이틀
st.title("My PDF Reader")

# PDF 파일 업로드
uploaded_file = st.file_uploader(
    "Choose a file",
    type="pdf",
    key="uploaded_pdf_file",
    on_change=upload_pdf_file,
    label_visibility="collapsed",
)

# 앱 재실행시 채팅 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 메시지 컨테이너
if prompt := st.chat_input("Please enter your message"):
    # 채팅 메시지 컨테이너에 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # 채팅 히스토리에 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 채팅 메시지 컨테이너에 GPT 메시지 표시
    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(prompt))  # AI 답변 생성

    # 채팅 히스토리에 GPT 메시지 추가
    st.session_state.messages.append({"role": "assistant", "content": response})
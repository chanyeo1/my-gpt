import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage

from modules.my_gpt.model import get_model

# 채팅 히스토리 초기화
if "store" not in st.session_state:
    st.session_state.store = {}

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("My ChatGPT")


# 세션 아이디(session_id)를 받아서 메세지 히스토리 오브젝트를 반환
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]


# 채팅 체인 생성
def create_chat_chain():
    # 프롬프트
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # GPT 모델
    llm = get_model(
        model_id=st.session_state._model_id,
        model_api_key=st.session_state._model_api_key,
    )

    # 체인 생성
    chat_chain = prompt | llm

    # 메시지 히스토리 조회 단계 추가
    chat_chain_with_history = RunnableWithMessageHistory(
        chat_chain,
        get_session_history,
    )

    return chat_chain_with_history


# 세션정보
config = {"configurable": {"session_id": "abc2"}}


# AI 답변 생성
def generate_response(human_message):
    chain = create_chat_chain()  # 체인 생성

    # 답변 생성
    # response = chain.invoke(
    #     [HumanMessage(content=human_message)],
    #     config=config
    # )

    return chain.stream([HumanMessage(content=human_message)], config=config)


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

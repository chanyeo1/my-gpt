import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_upstage import ChatUpstage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage

# 채팅 히스토리 초기화
if "store" not in st.session_state:
    st.session_state.store = {}

if "messages" not in st.session_state:
    st.session_state["messages"] = []


st.title("My ChatGPT")


# 설정 페이지에서 선택한 모델을 반환
def get_model():
    model_id = st.session_state._model_id
    model = None
    if model_id == 0:  # OPENAI
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=st.session_state._model_api_key,
        )
    elif model_id == 1:  # ANTHROPIC
        model = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=2,
            api_key=st.session_state._model_api_key,
        )
    elif model_id == 2:  # COHERE
        model = ChatCohere(cohere_api_key=st.session_state._model_api_key)
    elif model_id == 3:  # UPSTAGE
        model = ChatUpstage(upstage_api_key=st.session_state._model_api_key)
    elif model_id == 4:  # GEMINI
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=st.session_state._model_api_key,
        )

    return model


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
    llm = get_model()

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
